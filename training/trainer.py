"""Reusable PyTorch training utilities."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.metrics import EvaluationResult, evaluate


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    epoch_time_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    history: list[EpochResult]
    best_validation_accuracy: float
    best_checkpoint_path: str | None


class EarlyStopping:
    """Track validation improvement and signal when training should stop."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def get_default_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    scaler: torch.amp.GradScaler | None = None,
    use_mixed_precision: bool = False,
) -> EvaluationResult:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    device = torch.device(device)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=use_mixed_precision and device.type == "cuda",
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return EvaluationResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
) -> None:
    """Save a training checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": validation_accuracy,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Load a checkpoint into a model and optionally an optimizer."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def append_epoch_to_csv(path: str | Path, result: EpochResult) -> None:
    """Append one epoch result to a CSV log file."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(result).keys()))
        if should_write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str | None = None,
    use_mixed_precision: bool = True,
    early_stopping_patience: int | None = 5,
    checkpoint_path: str | Path | None = "checkpoints/best_model.pt",
    log_csv_path: str | Path | None = "results/training_log.csv",
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> TrainingResult:
    """Train a model with validation, checkpointing, early stopping, and CSV logs."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    device = get_default_device() if device is None else torch.device(device)
    model.to(device)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=use_mixed_precision and device.type == "cuda",
    )
    early_stopping = (
        EarlyStopping(patience=early_stopping_patience)
        if early_stopping_patience is not None
        else None
    )

    history: list[EpochResult] = []
    best_validation_accuracy = float("-inf")
    best_checkpoint_path: str | None = None

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
        )
        validation_result = evaluate(model, validation_loader, criterion, device)
        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=train_result.loss,
            train_accuracy=train_result.accuracy,
            validation_loss=validation_result.loss,
            validation_accuracy=validation_result.accuracy,
            epoch_time_seconds=time.perf_counter() - start_time,
        )
        history.append(epoch_result)

        if log_csv_path is not None:
            append_epoch_to_csv(log_csv_path, epoch_result)

        if validation_result.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_result.accuracy
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    validation_result.accuracy,
                )
                best_checkpoint_path = str(checkpoint_path)

        tqdm.write(
            "epoch "
            f"{epoch}/{epochs} "
            f"train_loss={train_result.loss:.4f} "
            f"train_acc={train_result.accuracy:.4f} "
            f"val_loss={validation_result.loss:.4f} "
            f"val_acc={validation_result.accuracy:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if early_stopping is not None and early_stopping.step(
            validation_result.accuracy
        ):
            break

    return TrainingResult(
        history=history,
        best_validation_accuracy=best_validation_accuracy,
        best_checkpoint_path=best_checkpoint_path,
    )
