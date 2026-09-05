// SPDX-License-Identifier: Apache-2.0
// Local standalone registration for the upstream persistent_topk operator.
#include <torch/csrc/stable/library.h>
#include "torch_utils.h"

void persistent_topk_det(const torch::stable::Tensor&, const torch::stable::Tensor&,
                        torch::stable::Tensor&, torch::stable::Tensor&,
                        int64_t, int64_t);

STABLE_TORCH_LIBRARY(_C_det, m) {
  m.def("persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
        "Tensor workspace, int k, int max_seq_len) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C_det, CUDA, m) {
  m.impl("persistent_topk", TORCH_BOX(&persistent_topk_det));
}
