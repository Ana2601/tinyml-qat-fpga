#!/usr/bin/env python3
"""TinyML quantization-aware-training sweep on M1.

Trains a small classifier on the synthetic tiny-image task at FP32 and with
QAT at 8 / 4 / 2-bit weight+activation quantization, then reports the
accuracy-vs-bitwidth trade-off -- the core experiment of the project.

The FPGA-host step (FINN / hls4ml dataflow deployment and on-board latency
measurement) lives in README.md; everything here is measured on M1.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

from fp_quant import QuantizedLinear, fresh_params, model_size_bits
from tiny_data import H, N_CLASSES, make_dataset

EPOCHS = 60
SEED = 0
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def build_model(w_bits: int = 32, a_bits: int = 32) -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Flatten(),
        QuantizedLinear(H * H, 64, w_bits=w_bits, a_bits=a_bits),
        nn.ReLU(),
        QuantizedLinear(64, N_CLASSES, w_bits=w_bits, a_bits=a_bits),
    )


def train(model, x, y, epochs) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x)
        l = loss(out, y)
        l.backward()
        opt.step()


def evaluate(model, x, y) -> float:
    with torch.no_grad():
        pred = model(x).argmax(-1)
    return float((pred == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", type=str, default=os.path.join(RESULTS, "sweep.json"))
    args = ap.parse_args()

    x_tr, y_tr, x_te, y_te = make_dataset()

    rows = []
    for bits in (32, 8, 4, 2):          # 32 == FP32 baseline
        w_bits = a_bits = bits
        model = build_model(w_bits, a_bits)
        train(model, x_tr, y_tr, args.epochs)
        acc = evaluate(model, x_te, y_te)
        size = model_size_bits(bits, fresh_params(model))
        rows.append({"bits": bits, "test_acc": round(acc, 4), "size_bits": size})
        tag = "FP32" if bits == 32 else f"QAT {bits}-bit"
        print(f"  {tag:>10} : test accuracy = {acc:.4f}   size = {size:,} bits")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"epochs": EPOCHS, "rows": rows}, f, indent=2)

    # summary: accuracy drop vs baseline
    base = rows[0]["test_acc"]
    print("\naccuracy drop vs FP32:")
    for r in rows[1:]:
        print(f"  {r['bits']:>2}-bit QAT : {-100.0*(r['test_acc']-base):.2f} pp  "
              f"({-100.0*(r['test_acc']-base)/max(base,1e-9):.2f}%)  "
              f"size/FP32 = {r['size_bits']/rows[0]['size_bits']:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())