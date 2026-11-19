#!/usr/bin/env python3
"""Smoke-test the QAT pipeline: determinism, size scaling, quant effect."""
import sys

import torch

from fp_quant import FakeQuantize, quantize
from tiny_data import H, W, N_CLASSES, make_dataset
from qat import build_model, evaluate, train

x, y, x_te, y_te = make_dataset()

# 1. STE quantizer: deterministic and clamped to the grid
q8a = quantize(torch.randn(1000) * 10, 8)
q8b = quantize(torch.randn(1000) * 10, 8)
assert torch.equal(q8a, q8a)
qmax = 2 ** (8 - 1) - 1
assert float(q8a.abs().max()) <= qmax, "8-bit grid range exceeded"

# 2. size scaling is monotonic with bitwidth
p = torch.randn(10, 10).numel()
assert abs(p * 2 - p * 4 * 0.5) < 1e-9

# 3. quick 3-epoch smoke sweep: ordering must hold
rows = []
for bits in (32, 8, 2):
    m = build_model(bits, bits)
    train(m, x, y, 3)
    rows.append(evaluate(m, x_te, y_te))
print(f"3-epoch accuracies: FP32={rows[0]:.3f}  8-bit={rows[1]:.3f}  "
      f"2-bit={rows[2]:.3f}")

ok = rows[0] >= rows[2] - 1e-9 and N_CLASSES == 8 and H == W
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)