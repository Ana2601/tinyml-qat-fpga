# TinyML on FPGA: Quantized Edge Inference (Brevitas-style QAT)

[![CI](https://github.com/Ana2601/tinyml-qat-fpga/actions/workflows/ci.yml/badge.svg)](https://github.com/Ana2601/tinyml-qat-fpga/actions/workflows/ci.yml)

The opposite end of datacenter HBM acceleration: **power- and
resource-constrained edge inference at sub-8-bit precision.** This is a quantization-aware-training (QAT) + accuracy-vs-bitwidth
study that runs end-to-end on M1; the FINN / hls4ml dataflow deployment
completes on the FPGA host.

## What runs on M1 (this repo)

A small classifier on a synthetic "tiny image" task (8 classes = quadrant
position x brightness) is trained FP32 and with QAT at **8 / 4 / 2-bit** using
a straight-through-estimator symmetric fake-quantizer — the same mechanism
Brevitas uses before exporting to QONNX/ONNX. The result is an honest
accuracy-vs-bitwidth sweep plus the parameter-size payoff.

Measured (60 epochs, deterministic seed, test split):

| model     | test acc | vs FP32 | params size | size/FP32 |
|-----------|---------:|--------:|------------:|----------:|
| FP32      | 0.684    | —       | 149,760 b   | 1.000x    |
| QAT 8-bit | 0.691    | +0.8 pp |  37,440 b   | 0.250x    |
| QAT 4-bit | 0.656    | −2.7 pp |  18,720 b   | 0.125x    |
| QAT 2-bit | 0.297    | −38.7 pp|   9,360 b   | 0.062x    |

Take-away, matching the published QAT pattern: **8-bit is effectively free,
4-bit costs ~4%, 2-bit falls off a cliff.** Accuracy alone says "ship 8-bit or
4-bit"; the FPGA-side study below decides latency/LUT vs. accuracy.

## FPGA host (next step)

- **hls4ml vs FINN trade-off study** on the same model:
  - hls4ml: HLS streaming implementation, typically lower latency / higher
    throughput, manual design control.
  - FINN: automated Brevitas->QONNX->dataflow, up to ~80% fewer LUT/FF for the
    same bitwidth.
- Both call Vitis HLS (x86-64 host), so this lands as a build-stage-only
  "prepare on M1, synthesize + measure on FPGA host".
- Target: a constrained edge board (PYNQ-Z2 class), not Alveo.

## Files

```
tiny_data.py    synthetic 8-class tiny-image dataset (offline, deterministic)
fp_quant.py     STE symmetric fake-quantizer (Brevitas-equivalent QAT)
qat.py          QAT sweep: FP32 / 8 / 4 / 2-bit, prints + saves results/sweep.json
test_qat.py     self-test: deterministic quantizer, size scaling, accuracy order
Makefile        demo / test
```

## Run

```bash
python3 qat.py        # full QAT sweep (torch; ~1 min on M1)
make test             # smoke self-test
```

Requires `torch` (numpy included). The dataset is generated offline in
`data/`.

## Setup

- PyTorch (CPU is enough)
- Python packages: `pip install -r requirements.txt`

```bash
python3 test_qat.py
```

The same commands run in CI on every push (`.github/workflows/ci.yml`).
