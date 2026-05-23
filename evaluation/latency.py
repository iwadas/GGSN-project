"""Hardware latency measurement utilities."""

from __future__ import annotations

import time

import torch
from torch import nn


@torch.inference_mode()
def measure_inference_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    device: torch.device | str | None = None,
    warmup_steps: int = 20,
    measured_steps: int = 100,
) -> float:
    """Measure average forward-pass latency in milliseconds."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if measured_steps < 1:
        raise ValueError("measured_steps must be at least 1.")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)
    model.eval()

    sample = torch.randn(input_shape, device=device)
    for _ in range(warmup_steps):
        model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(measured_steps):
        model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / measured_steps
