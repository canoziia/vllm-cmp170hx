#!/usr/bin/env python3
"""Materialize DeepSeek-V4.1 Engram CPU tables into page-aligned shared memory shards.

Populates a shared directory (typically a host ``tmpfs`` mount such as
``/mnt/engram-shm``) with ``--num-shards`` (default 4) disjoint, 4 KiB
page-aligned shard files per Engram layer so that:
  - TP4 Decode ranks each ``mmap(MAP_SHARED)`` and ``cudaHostRegister`` one
    completely disjoint shard file (native GPU UVA speed, zero overlap);
  - TP2 Prefill ranks each ``mmap(MAP_SHARED, PROT_READ)`` two contiguous
    shard files without CUDA registration and read via multi-shard C++
    ``engram_cpu_gather_*_multi`` -> pinned staging -> async H2D DMA.

Both containers share the exact same 189.13 GiB of physical host RAM, and
restarting either container reuses the warm tmpfs tables in seconds.

Pure Python standard library (zero third-party dependencies on the host).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

PAGE_SIZE = 4096
COPY_CHUNK = 64 * 1024 * 1024  # 64 MiB streaming chunks


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    limit = math.isqrt(n)
    d = 5
    while d <= limit:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _find_next_prime(start: int, seen: set[int]) -> int:
    cand = max(2, start + 1)
    while cand in seen or not _is_prime(cand):
        cand += 1
    return cand


def _align_up(n: int, align: int = PAGE_SIZE) -> int:
    return (n + align - 1) & ~(align - 1)


def _load_text_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if isinstance(cfg.get("text_config"), dict) and "engram_layer_ids" not in cfg:
        return cfg["text_config"]
    return cfg


def _compute_engram_layout(cfg: dict) -> dict:
    layer_ids = [int(x) for x in cfg["engram_layer_ids"]]
    max_ngram = int(cfg["engram_max_ngram_size"])
    n_heads = int(cfg["engram_n_heads"])
    head_dim = int(cfg["engram_head_dim"])
    engram_vocab_size = int(cfg["engram_vocab_size"])
    quant = cfg.get("quantization_config") or {}
    embed_quant = quant.get("engram_embed_quant", "fp8")
    if embed_quant not in ("fp8", "mxfp4"):
        raise ValueError(f"Unsupported engram_embed_quant={embed_quant!r}")
    packed_fp4 = embed_quant == "mxfp4"
    block_size = int(quant.get("engram_embed_block_size", 32))
    row_bytes = head_dim // 2 if packed_fp4 else head_dim
    scale_dim = head_dim // block_size

    seen: set[int] = set()
    layer_head_sizes: dict[int, list[int]] = {}
    for lid in layer_ids:
        flat: list[int] = []
        for _ in range(max_ngram - 1):
            cur = engram_vocab_size - 1
            for _ in range(n_heads):
                cur = _find_next_prime(cur, seen)
                seen.add(cur)
                flat.append(cur)
        layer_head_sizes[lid] = flat

    return {
        "layer_ids": layer_ids,
        "head_dim": head_dim,
        "block_size": block_size,
        "packed_fp4": packed_fp4,
        "row_bytes": row_bytes,
        "scale_dim": scale_dim,
        "n_hash_cols": (max_ngram - 1) * n_heads,
        "layer_head_sizes": layer_head_sizes,
    }


def _find_engram_tensors(model_dir: Path) -> dict[str, tuple[Path, int, int, list[int], str]]:
    """Return map tensor_name -> (file_path, abs_data_start, byte_len, shape, dtype)."""
    found: dict[str, tuple[Path, int, int, list[int], str]] = {}
    for fn in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        p = Path(fn)
        with open(p, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        data_base = 8 + hdr_len
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if "engram.embed." in k:
                s, e = v["data_offsets"]
                found[k] = (p, data_base + int(s), int(e) - int(s), list(v["shape"]), v["dtype"])
    return found


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _copy_slice(
    src_path: Path,
    src_offset: int,
    exact_bytes: int,
    dst_path: Path,
) -> int:
    aligned_bytes = _align_up(exact_bytes, PAGE_SIZE)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    src_fd = os.open(src_path, os.O_RDONLY)
    dst_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
            try:
                os.posix_fadvise(src_fd, src_offset, exact_bytes, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
        os.ftruncate(dst_fd, aligned_bytes)
        copied = 0
        while copied < exact_bytes:
            want = min(COPY_CHUNK, exact_bytes - copied)
            buf = os.pread(src_fd, want, src_offset + copied)
            if not buf:
                raise EOFError(
                    f"Short read from {src_path} at {src_offset + copied}: got 0 of {want}"
                )
            view = memoryview(buf)
            written = 0
            while written < len(view):
                n = os.pwrite(dst_fd, view[written:], copied + written)
                written += n
            _fadvise_dontneed(src_fd, src_offset + copied, len(buf))
            copied += len(buf)
        if aligned_bytes > exact_bytes:
            pad = bytes(aligned_bytes - exact_bytes)
            os.pwrite(dst_fd, pad, exact_bytes)
    except Exception:
        os.close(dst_fd)
        os.close(src_fd)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    os.close(dst_fd)
    os.close(src_fd)
    os.replace(tmp_path, dst_path)
    return aligned_bytes


def _is_manifest_valid(
    out_dir: Path,
    expected_manifest: dict,
) -> bool:
    mpath = out_dir / "manifest.json"
    if not mpath.is_file():
        return False
    try:
        existing = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return False
    if existing != expected_manifest:
        return False
    for lmeta in existing.get("layers", {}).values():
        for s in lmeta.get("shards", []):
            wf = out_dir / s["weight_file"]
            sf = out_dir / s["scale_file"]
            if not wf.is_file() or not sf.is_file():
                return False
            if wf.stat().st_size != int(s["weight_file_bytes"]):
                return False
            if sf.stat().st_size != int(s["scale_file_bytes"]):
                return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to DeepSeek-V4.1 checkpoint")
    ap.add_argument("--out-dir", required=True, help="Target shared tmpfs directory")
    ap.add_argument("--num-shards", type=int, default=4, help="Disjoint shards per layer (default: 4)")
    ap.add_argument("--force", action="store_true", help="Rebuild even if manifest matches")
    ap.add_argument("--dry-run", action="store_true", help="Compute layout and print plan only")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir)
    num_shards = int(args.num_shards)

    cfg = _load_text_config(model_dir)
    layout = _compute_engram_layout(cfg)
    tensors = _find_engram_tensors(model_dir)
    if layout["n_hash_cols"] % num_shards != 0:
        raise ValueError(
            f"n_hash_cols={layout['n_hash_cols']} must be divisible by num_shards={num_shards}"
        )
    heads_per_shard = layout["n_hash_cols"] // num_shards

    manifest: dict = {
        "version": 1,
        "model_dir": str(model_dir),
        "num_shards": num_shards,
        "layers": {},
    }
    copy_jobs: list[tuple[Path, int, int, Path]] = []
    total_bytes = 0

    for lid in layout["layer_ids"]:
        w_key = f"layers.{lid}.engram.embed.weight"
        s_key = f"layers.{lid}.engram.embed.scale"
        if w_key not in tensors:
            w_key = f"model.{w_key}"
            s_key = f"model.{s_key}"
        w_file, w_off, w_len, w_shape, _ = tensors[w_key]
        s_file, s_off, s_len, s_shape, _ = tensors[s_key]
        head_sizes = layout["layer_head_sizes"][lid]
        assert sum(head_sizes) <= w_shape[0], (
            f"layer {lid}: sum(head_sizes)={sum(head_sizes)} > tensor rows {w_shape[0]}"
        )
        assert w_shape[1] == layout["row_bytes"]
        assert s_shape[1] == layout["scale_dim"]

        shards_meta = []
        for sid in range(num_shards):
            h0 = sid * heads_per_shard
            h1 = h0 + heads_per_shard
            v0 = sum(head_sizes[:h0])
            v1 = sum(head_sizes[:h1])
            rows = v1 - v0
            w_exact = rows * layout["row_bytes"]
            s_exact = rows * layout["scale_dim"]
            w_aligned = _align_up(w_exact)
            s_aligned = _align_up(s_exact)
            wf_name = f"layer_{lid}_s{sid}_w.bin"
            sf_name = f"layer_{lid}_s{sid}_s.bin"
            shards_meta.append(
                {
                    "shard_id": sid,
                    "vocab_start": v0,
                    "vocab_end": v1,
                    "weight_file": wf_name,
                    "scale_file": sf_name,
                    "weight_exact_bytes": w_exact,
                    "scale_exact_bytes": s_exact,
                    "weight_file_bytes": w_aligned,
                    "scale_file_bytes": s_aligned,
                }
            )
            copy_jobs.append((w_file, w_off + v0 * layout["row_bytes"], w_exact, out_dir / wf_name))
            copy_jobs.append((s_file, s_off + v0 * layout["scale_dim"], s_exact, out_dir / sf_name))
            total_bytes += w_aligned + s_aligned

        w_stat = w_file.stat()
        manifest["layers"][str(lid)] = {
            "dim": layout["head_dim"],
            "block_size": layout["block_size"],
            "packed_fp4": layout["packed_fp4"],
            "total_rows": sum(head_sizes),
            "source_file": w_file.name,
            "source_size": w_stat.st_size,
            "source_mtime_ns": w_stat.st_mtime_ns,
            "shards": shards_meta,
        }

    print(
        f"Engram SHM plan: layers={layout['layer_ids']}, num_shards={num_shards}, "
        f"total={total_bytes / 1024**3:.2f} GiB -> {out_dir}"
    )
    for lid_str, lmeta in manifest["layers"].items():
        for s in lmeta["shards"]:
            print(
                f"  layer {lid_str} shard {s['shard_id']}: rows=[{s['vocab_start']}, {s['vocab_end']}) "
                f"({s['vocab_end'] - s['vocab_start']} rows, "
                f"{(s['weight_file_bytes'] + s['scale_file_bytes']) / 1024**3:.2f} GiB)"
            )

    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and _is_manifest_valid(out_dir, manifest):
        print("Engram SHM manifest & all shards already valid — skipping copy.")
        return 0

    t0 = time.monotonic()
    for src_path, src_off, exact_bytes, dst_path in copy_jobs:
        ts = time.monotonic()
        _copy_slice(src_path, src_off, exact_bytes, dst_path)
        dt = time.monotonic() - ts
        bw = (exact_bytes / 1024**3) / max(dt, 1e-6)
        print(
            f"  wrote {dst_path.name}: {exact_bytes / 1024**3:.2f} GiB "
            f"in {dt:.1f}s ({bw:.2f} GiB/s)",
            flush=True,
        )

    tmp_manifest = out_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, out_dir / "manifest.json")
    print(f"Done in {time.monotonic() - t0:.1f}s: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
