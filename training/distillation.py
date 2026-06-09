"""Knowledge Distillation utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean")
    kd_loss = kd_loss * (temperature**2)

    ce_loss = criterion(student_logits, targets)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


@torch.inference_mode()
def precompute_teacher_logits(
    teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    teacher.eval()
    teacher.to(device)
    all_logits: list[torch.Tensor] = []

    for inputs, _ in tqdm(dataloader, desc="precompute teacher logits", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        all_logits.append(teacher(inputs).cpu())

    return torch.cat(all_logits, dim=0)


def distill_epoch(
    student: nn.Module,
    teacher_logits: torch.Tensor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
    max_grad_norm: float | None = 5.0,
) -> float:
    student.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    logit_offset = 0

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        batch_size = targets.size(0)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        teacher_logits_batch = teacher_logits[logit_offset : logit_offset + batch_size].to(device, non_blocking=True)
        logit_offset += batch_size

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(inputs)
        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits_batch,
            targets=targets,
            temperature=temperature,
            alpha=alpha,
        )
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_correct += (student_logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples
