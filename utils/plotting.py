"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_training_curves(
    log_csv_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot train/validation loss and accuracy curves from a training CSV."""
    import matplotlib.pyplot as plt

    history = pd.read_csv(log_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(
        history["epoch"],
        history["validation_accuracy"],
        label="validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
