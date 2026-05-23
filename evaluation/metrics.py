"""Common model evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.latency import measure_inference_latency


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

