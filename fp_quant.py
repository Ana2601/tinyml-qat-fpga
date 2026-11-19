"""Straight-through-estimator (STE) symmetric fake-quantization.

Mirrors quantization-aware training as done by Brevitas/pytorch-quantization:
the forward pass quantizes to b-bit integers, the backward pass passes
gradients through unchanged (STE). We use per-tensor symmetric quantization,
the default used when exporting to FINN/hls4ml edge targets.
"""
import torch
import torch.nn as nn


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 32:
            return x.contiguous()
        qmax = float(2 ** (bits - 1) - 1)
        scale = x.abs().max().clamp_min(1e-8).item() / qmax
        xq = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
        return xq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    return FakeQuantize.apply(x, bits)


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, w_bits: int = 32, a_bits: int = 32, **kw):
        super().__init__(*args, **kw)
        self.w_bits, self.a_bits = w_bits, a_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            quantize(x, self.a_bits), quantize(self.weight, self.w_bits),
            quantize(self.bias, self.w_bits) if self.bias is not None else None,
        )


def model_size_bits(bits: int, params: dict[str, int]) -> int:
    """Parameter bit-weight of the model at the given weight bit-width."""
    return sum(n * bits for n in params.values())


def fresh_params(model: nn.Module) -> dict[str, int]:
    return {k: v.numel() for k, v in model.state_dict().items()}