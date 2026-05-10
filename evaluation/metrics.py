"""Common model evaluation metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return batch accuracy as a float in the range [0, 1]."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> EvaluationResult:
    """Evaluate loss and accuracy over a dataloader."""
    model.eval()
    device = torch.device(device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    return EvaluationResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


@torch.inference_mode()
def measure_inference_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    device: torch.device | str | None = None,
    warmup_steps: int = 20,
    measured_steps: int = 100,
) -> float:
    """Measure average forward-pass latency in milliseconds."""
    if measured_steps < 1:
        raise ValueError("measured_steps must be at least 1.")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
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
