#!/usr/bin/env python3
"""Generate the provisioned Grafana dashboards.

* vllm-throughput.json: our throughput overview (written from scratch).
* vllm-official.json / vllm-performance.json: upstream vLLM dashboards
  (tools/upstream/, vllm-project/vllm examples/observability) with
  server/instance/role variables injected into every vllm:* selector, so
  several instances serving the same model_name are no longer merged.

Run after editing:  python3 monitoring/tools/gen_dashboards.py
"""
import copy
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
F = 'job="vllm",server=~"$server",instance=~"$instance",role=~"$role"'
RI = "$__rate_interval"


# --------------------------------------------------------------- variables
def ds_var():
    return {"name": "DS_PROMETHEUS", "label": "Datasource", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 0, "regex": "",
            "refresh": 1}


def q_var(name, label, query, include_all=True):
    return {"name": name, "label": label, "type": "query", "datasource": DS,
            "definition": query, "query": {"query": query, "refId": name},
            "refresh": 2, "multi": True, "includeAll": include_all,
            "allValue": ".*", "current": {"text": ["All"], "value": ["$__all"]},
            "sort": 1, "hide": 0, "options": [], "regex": ""}


def common_vars():
    return [
        ds_var(),
        q_var("server", "server", 'label_values(vllm:num_requests_running{job="vllm"}, server)'),
        q_var("role", "role", 'label_values(vllm:num_requests_running{job="vllm",server=~"$server"}, role)'),
        q_var("instance", "instance",
              'label_values(vllm:num_requests_running{job="vllm",server=~"$server",role=~"$role"}, instance)'),
    ]


# ------------------------------------------------------------------ panels
_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def tgt(expr, legend="", ref="A", instant=False, interval=None, fmt=None):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref,
         "range": not instant, "instant": instant}
    if fmt:
        t["format"] = fmt
    if interval:
        t["interval"] = interval
    return t


def ts(title, targets, x, y, w=12, h=8, unit="short", desc="", stack=False,
       bars=False, decimals=None, minv=0, maxv=None):
    defaults = {"unit": unit, "min": minv,
                "custom": {"drawStyle": "bars" if bars else "line",
                           "fillOpacity": 60 if bars else 10, "lineWidth": 1,
                           "showPoints": "never", "spanNulls": False,
                           "stacking": {"mode": "normal" if stack else "none"}}}
    if maxv is not None:
        defaults["max"] = maxv
    if decimals is not None:
        defaults["decimals"] = decimals
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc,
            "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [tgt(*t) if isinstance(t, tuple) else t for t in targets],
            "fieldConfig": {"defaults": defaults, "overrides": [
                {"matcher": {"id": "byName", "options": "total"},
                 "properties": [{"id": "custom.lineWidth", "value": 3},
                                {"id": "custom.fillOpacity", "value": 0},
                                {"id": "custom.stacking", "value": {"mode": "none"}}]}]},
            "options": {"legend": {"displayMode": "table", "placement": "right",
                                   "calcs": ["mean", "max", "lastNotNull"]},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def stat(title, expr, x, y, w=4, h=4, unit="short", desc="", decimals=None,
         instant=False):
    d = {"unit": unit, "color": {"mode": "fixed", "fixedColor": "blue"}}
    if decimals is not None:
        d["decimals"] = decimals
    return {"id": nid(), "type": "stat", "title": title, "description": desc,
            "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [tgt(expr, "", "A", instant)],
            "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                        "colorMode": "value", "graphMode": "area",
                        "textMode": "value"}}


def row(title, y):
    return {"id": nid(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def text(content, x, y, w=24, h=4):
    return {"id": nid(), "type": "text", "title": "口径 / Accounting",
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": content}}


def table(title, targets, x, y, w=24, h=8, unit="short"):
    return {"id": nid(), "type": "table", "title": title, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": targets,
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "transformations": [
                {"id": "merge"},
                {"id": "organize", "options": {
                    "excludeByName": {"Time": True},
                    "renameByName": {"Value #A": "generation tokens",
                                     "Value #B": "prompt tokens",
                                     "Value #C": "prompt computed locally",
                                     "Value #D": "prompt from local prefix cache",
                                     "Value #E": "prompt from external KV",
                                     "Value #F": "finished requests"}}}],
            "options": {"showHeader": True, "footer": {"show": True,
                        "reducer": ["sum"], "fields": ""}}}


GEN = f'vllm:generation_tokens_total{{{F},role!="prefill"}}'
PROMPT = f'vllm:prompt_tokens_total{{{F},role!="decode"}}'
REQ = f'vllm:request_success_total{{{F},role!="prefill"}}'

NOTE = """**PD 口径（避免重复计数）**：每个目标带 `role` 标签（`prefill` / `decode` / `unified`，由 `VLLM_METRICS_TARGETS` 名称推断或显式 `;role=`）。
PD 对中 prefill 只生成 1 个 token、decode 会再次计入同一 prompt，因此：**生成 tokens/请求数 = 排除 role=prefill**；**用户 prompt tokens = 排除 role=decode**；
`prompt 来源` 按 `vllm:prompt_tokens_by_source_total` 原样分 `local_compute`（真正计算）/ `local_cache_hit`（GPU 前缀缓存）/ `external_kv_transfer`（LMCache 或 PD 传输），这些是各实例实际工作量，PD 对中 decode 的 external 部分即 KV 传输量。
spec decode：接受率 = accepted / draft tokens；每步平均产出 = 1 + accepted / drafts（含目标模型自身 1 token）。"""


def throughput():
    _id[0] = 0
    p = []
    y = 0
    p.append(text(NOTE, 0, y)); y += 4
    p.append(row("总计 (按所选 server/role/instance 过滤)", y)); y += 1
    p += [
        stat("generation tokens/s", f"sum(rate({GEN}[{RI}]))", 0, y, unit="none", decimals=1,
             desc="sum over role!=prefill"),
        stat("user prompt tokens/s", f"sum(rate({PROMPT}[{RI}]))", 4, y, unit="none", decimals=1,
             desc="sum over role!=decode"),
        stat("prompt computed tokens/s",
             f'sum(rate(vllm:prompt_tokens_by_source_total{{{F},source="local_compute"}}[{RI}]))',
             8, y, unit="none", decimals=1, desc="all roles, source=local_compute"),
        stat("finished requests/s", f"sum(rate({REQ}[{RI}]))", 12, y, unit="reqps", decimals=2),
        stat("running", f'sum(vllm:num_requests_running{{{F}}})', 16, y, w=2, decimals=0),
        stat("waiting", f'sum(vllm:num_requests_waiting{{{F}}})', 18, y, w=2, decimals=0),
        stat("targets up", f'sum(up{{{F}}}) / count(up{{{F}}})', 20, y, unit="percentunit", decimals=0),
    ]
    y += 4
    p += [
        stat("generation tokens (range)", f"sum(increase({GEN}[$__range]))", 0, y, unit="short",
             instant=True, desc="dashboard time range"),
        stat("user prompt tokens (range)", f"sum(increase({PROMPT}[$__range]))", 4, y, unit="short",
             instant=True),
        stat("prompt cached tokens (range)",
             f'sum(increase(vllm:prompt_tokens_cached_total{{{F},role!="decode"}}[$__range]))',
             8, y, unit="short", instant=True, desc="local + external cache, role!=decode"),
        stat("finished requests (range)", f"sum(increase({REQ}[$__range]))", 12, y, instant=True),
        stat("GPU prefix-cache hit (range)",
             f'sum(increase(vllm:prefix_cache_hits_total{{{F}}}[$__range])) / sum(increase(vllm:prefix_cache_queries_total{{{F}}}[$__range]))',
             16, y, unit="percentunit", instant=True, decimals=1),
        stat("external (LMCache) hit (range)",
             f'sum(increase(vllm:external_prefix_cache_hits_total{{{F}}}[$__range])) / sum(increase(vllm:external_prefix_cache_queries_total{{{F}}}[$__range]))',
             20, y, unit="percentunit", instant=True, decimals=1),
    ]
    y += 4
    p.append(row("吞吐 (各实例 + total)", y)); y += 1
    p += [
        ts("generation tokens/s", [
            (f"sum by (instance) (rate(vllm:generation_tokens_total{{{F}}}[{RI}]))", "{{instance}}", "A"),
            (f"sum(rate({GEN}[{RI}]))", "total", "B")], 0, y, unit="none",
           desc="per instance (all roles); total excludes role=prefill"),
        ts("prompt tokens/s", [
            (f"sum by (instance) (rate(vllm:prompt_tokens_total{{{F}}}[{RI}]))", "{{instance}}", "A"),
            (f"sum(rate({PROMPT}[{RI}]))", "total", "B")], 12, y, unit="none",
           desc="per instance (all roles); total excludes role=decode"),
    ]
    y += 8
    p += [
        ts("prompt tokens/s by source", [
            (f"sum by (instance, source) (rate(vllm:prompt_tokens_by_source_total{{{F}}}[{RI}]))",
             "{{instance}} {{source}}", "A")], 0, y, unit="none", stack=True),
        ts("finished requests/s", [
            (f"sum by (instance) (rate(vllm:request_success_total{{{F}}}[{RI}]))", "{{instance}}", "A"),
            (f"sum(rate({REQ}[{RI}]))", "total", "B"),
            (f'sum by (finished_reason) (rate({REQ[:-1]},finished_reason!~"stop|length"}}[{RI}]))',
             "total {{finished_reason}}", "C")], 12, y, unit="reqps"),
    ]
    y += 8
    p += [
        ts("running / waiting requests", [
            (f"sum by (instance) (vllm:num_requests_running{{{F}}})", "{{instance}} running", "A"),
            (f"sum by (instance) (vllm:num_requests_waiting{{{F}}})", "{{instance}} waiting", "B")],
           0, y, decimals=0),
        ts("KV cache usage", [
            (f"max by (instance) (vllm:kv_cache_usage_perc{{{F}}})", "{{instance}}", "A")],
           12, y, unit="percentunit", maxv=1),
    ]
    y += 8
    p.append(row("缓存命中", y)); y += 1
    p += [
        ts("GPU prefix-cache hit rate", [
            (f"sum by (instance) (rate(vllm:prefix_cache_hits_total{{{F}}}[{RI}])) / sum by (instance) (rate(vllm:prefix_cache_queries_total{{{F}}}[{RI}]))",
             "{{instance}}", "A")], 0, y, unit="percentunit", maxv=1),
        ts("external (LMCache / KV connector) hit rate", [
            (f"sum by (instance) (rate(vllm:external_prefix_cache_hits_total{{{F}}}[{RI}])) / sum by (instance) (rate(vllm:external_prefix_cache_queries_total{{{F}}}[{RI}]))",
             "{{instance}}", "A")], 12, y, unit="percentunit", maxv=1),
    ]
    y += 8
    p.append(row("延迟 (各实例)", y)); y += 1
    for i, (name, metric) in enumerate([("TTFT", "time_to_first_token_seconds"),
                                         ("ITL (inter-token)", "inter_token_latency_seconds"),
                                         ("E2E request latency", "e2e_request_latency_seconds"),
                                         ("queue time", "request_queue_time_seconds")]):
        p.append(ts(f"{name} p50 / p90 / p99", [
            (f"histogram_quantile({q}, sum by (le, instance) (rate(vllm:{metric}_bucket{{{F}}}[{RI}])))",
             f"{{{{instance}}}} p{int(float(q)*100)}", "ABC"[k])
            for k, q in enumerate(["0.5", "0.9", "0.99"])], (i % 2) * 12, y, unit="s"))
        if i % 2:
            y += 8
    p.append(row("Speculative decoding (MTP / DFlash / DSpark)", y)); y += 1
    p += [
        ts("draft acceptance rate", [
            (f"sum by (instance) (rate(vllm:spec_decode_num_accepted_tokens_total{{{F}}}[{RI}])) / sum by (instance) (rate(vllm:spec_decode_num_draft_tokens_total{{{F}}}[{RI}]))",
             "{{instance}}", "A")], 0, y, w=8, unit="percentunit", maxv=1),
        ts("tokens per decode step (1 + accepted/drafts)", [
            (f"1 + sum by (instance) (rate(vllm:spec_decode_num_accepted_tokens_total{{{F}}}[{RI}])) / sum by (instance) (rate(vllm:spec_decode_num_drafts_total{{{F}}}[{RI}]))",
             "{{instance}}", "A")], 8, y, w=8, decimals=2),
        ts("acceptance by draft position", [
            (f"sum by (instance, position) (rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{{{F}}}[{RI}])) / on (instance) group_left sum by (instance) (rate(vllm:spec_decode_num_drafts_total{{{F}}}[{RI}]))",
             "{{instance}} pos {{position}}", "A")], 16, y, w=8, unit="percentunit", maxv=1),
    ]
    y += 8
    p.append(row("累计 token 统计", y)); y += 1
    p += [
        ts("generation tokens per hour", [
            tgt(f"sum by (instance) (increase({GEN}[1h]))", "{{instance}}", "A", interval="1h")],
           0, y, w=8, unit="short", stack=True, bars=True, desc="role!=prefill"),
        ts("generation tokens per day (last 30d)", [
            tgt(f"sum by (instance) (increase({GEN}[1d]))", "{{instance}}", "A", interval="1d")],
           8, y, w=8, unit="short", stack=True, bars=True,
           desc="fixed 30d window, 1d steps; role!=prefill"),
        ts("user prompt tokens per day (last 30d)", [
            tgt(f"sum by (instance) (increase({PROMPT}[1d]))", "{{instance}}", "A", interval="1d")],
           16, y, w=8, unit="short", stack=True, bars=True,
           desc="fixed 30d window, 1d steps; role!=decode"),
    ]
    p[-2]["timeFrom"] = p[-1]["timeFrom"] = "30d"
    y += 8
    p.append(table("区间累计 (dashboard time range, 各实例)", [
        tgt(f"sum by (instance, role) (increase(vllm:generation_tokens_total{{{F}}}[$__range]))", "", "A", True, fmt="table"),
        tgt(f"sum by (instance, role) (increase(vllm:prompt_tokens_total{{{F}}}[$__range]))", "", "B", True, fmt="table"),
        tgt(f'sum by (instance, role) (increase(vllm:prompt_tokens_by_source_total{{{F},source="local_compute"}}[$__range]))', "", "C", True, fmt="table"),
        tgt(f'sum by (instance, role) (increase(vllm:prompt_tokens_by_source_total{{{F},source="local_cache_hit"}}[$__range]))', "", "D", True, fmt="table"),
        tgt(f'sum by (instance, role) (increase(vllm:prompt_tokens_by_source_total{{{F},source="external_kv_transfer"}}[$__range]))', "", "E", True, fmt="table"),
        tgt(f"sum by (instance, role) (increase(vllm:request_success_total{{{F}}}[$__range]))", "", "F", True, fmt="table"),
    ], 0, y, h=9, unit="short"))
    y += 9
    p.append(row("vllm-router / LMCache / GPU (可选 job)", y)); y += 1
    p += [
        ts("LMCache lookup hit ratio (tokens)", [
            ('sum by (instance) (rate(lmcache_mp_lookup_hit_tokens_total{job="lmcache",server=~"$server"}[$__rate_interval])) / sum by (instance) (rate(lmcache_mp_lookup_requested_tokens_total{job="lmcache",server=~"$server"}[$__rate_interval]))',
             "{{instance}}", "A")], 0, y, w=8, unit="percentunit", maxv=1),
        ts("LMCache L1 usage", [
            ('max by (instance) (lmcache_mp_l1_usage_ratio{job="lmcache",server=~"$server"})', "{{instance}} L1 ratio", "A")],
           8, y, w=8, unit="percentunit", maxv=1),
        ts("router healthy pods / GPU prefix hit", [
            ('max by (instance) (vllm:healthy_pods_total{job="vllm-router",server=~"$server"})', "{{instance}} healthy pods", "A"),
            ('max by (instance) (vllm:gpu_prefix_cache_hit_rate{job="vllm-router",server=~"$server"})', "{{instance}} prefix hit", "B")],
           16, y, w=8),
    ]
    y += 8
    p += [
        ts("GPU utilization (nvidia-smi exporter)", [
            ('nvidia_smi_utilization_gpu_ratio{job="gpu",server=~"$server"}', "{{instance}} {{uuid}}", "A")],
           0, y, unit="percentunit", maxv=1),
        ts("GPU power", [
            ('nvidia_smi_power_draw_watts{job="gpu",server=~"$server"}', "{{instance}} {{uuid}}", "A")],
           12, y, unit="watt"),
    ]
    return {"uid": "vllm-throughput", "title": "vLLM 吞吐总览 (throughput overview)",
            "tags": ["vllm"], "timezone": "browser", "schemaVersion": 41,
            "editable": False, "graphTooltip": 1, "refresh": "30s",
            "time": {"from": "now-6h", "to": "now"},
            "templating": {"list": common_vars()}, "annotations": {"list": []},
            "panels": p}


# --------------------------------------------------------------- upstream
SEL = re.compile(r"(vllm:[a-zA-Z0-9_:]+)(\{([^}]*)\})?")


def inject(expr):
    def rep(m):
        inner = (m.group(3) or "").strip()
        inner = re.sub(r'model_name="\$model_name"', 'model_name=~"$model_name"', inner)
        inner = re.sub(r'model_name=~"\$Deployment_id"', 'model_name=~"$model_name"', inner)
        return f"{m.group(1)}{{{F}{',' + inner if inner else ''}}}"
    expr = SEL.sub(rep, expr)
    # keep instances apart in aggregations
    expr = re.sub(r"sum by ?\(le\)", "sum by (le, instance)", expr)
    expr = re.sub(r"sum by ?\(le, model_name\)", "sum by (le, instance, model_name)", expr)
    expr = re.sub(r"sum by ?\(finished_reason\)", "sum by (instance, finished_reason)", expr)
    return expr


def adapt(src, uid, title):
    d = json.loads((HERE / "upstream" / src).read_text())
    d = copy.deepcopy(d)
    for k in ("id", "__inputs", "__requires", "__elements"):
        d.pop(k, None)
    d["uid"], d["title"], d["editable"] = uid, title, False
    keep = [v for v in d.get("templating", {}).get("list", [])
            if v["type"] == "custom"]
    model = q_var("model_name", "model_name",
                  f'label_values(vllm:num_requests_running{{{F}}}, model_name)')
    d["templating"] = {"list": common_vars() + [model] + keep}

    def walk(o):
        if isinstance(o, dict):
            if "datasource" in o and isinstance(o["datasource"], dict):
                o["datasource"] = DS
            if isinstance(o.get("expr"), str) and "vllm:" in o["expr"]:
                o["expr"] = inject(o["expr"])
                lf = o.get("legendFormat", "") or ""
                if "{{instance}}" not in lf:
                    o["legendFormat"] = ("{{instance}} " + lf).strip() if lf not in ("", "__auto") else "{{instance}}"
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    d["tags"] = sorted(set(d.get("tags", [])) | {"vllm", "upstream"})
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {
        "vllm-throughput.json": throughput(),
        "vllm-official.json": adapt("grafana.json", "vllm-official", "vLLM (official, per instance)"),
        "vllm-performance.json": adapt("performance_statistics.json", "vllm-performance",
                                       "vLLM performance statistics (official, per instance)"),
    }
    for name, d in out.items():
        (OUT / name).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
