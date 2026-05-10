"""Run the baseline CNN CIFAR-10 experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from data.dataloader import get_cifar10_dataloaders
from evaluation.metrics import count_parameters, evaluate, measure_inference_latency
from models.baseline_cnn import build_baseline_cnn
from training.trainer import fit
from utils.config import load_yaml_config
from utils.plotting import plot_training_curves
from utils.reproducibility import set_seed


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run baseline CNN training, evaluation, and artifact generation."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]

    train_loader, validation_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=int(data_config["batch_size"]),
        validation_ratio=float(data_config["validation_ratio"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        download=bool(data_config["download"]),
    )

    model = build_baseline_cnn(
        num_layers=int(model_config["num_layers"]),
        base_filters=int(model_config["base_filters"]),
        filter_multiplier=int(model_config["filter_multiplier"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config["num_classes"]),
    )
    parameter_count = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    training_result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=int(training_config["epochs"]),
        use_mixed_precision=bool(training_config["use_mixed_precision"]),
        early_stopping_patience=int(training_config["early_stopping_patience"]),
        checkpoint_path=output_config["checkpoint_path"],
        log_csv_path=output_config["log_csv_path"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output_config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_result = evaluate(model, test_loader, criterion, device)
    latency_ms = measure_inference_latency(model, device=device)

    summary = {
        "seed": seed,
        "epochs": int(training_config["epochs"]),
        "batch_size": int(data_config["batch_size"]),
        "learning_rate": float(training_config["learning_rate"]),
        "num_layers": int(model_config["num_layers"]),
        "base_filters": int(model_config["base_filters"]),
        "dropout": float(model_config["dropout"]),
        "parameters": parameter_count,
        "best_validation_accuracy": training_result.best_validation_accuracy,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "latency_ms": latency_ms,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": output_config["checkpoint_path"],
        "log_csv_path": output_config["log_csv_path"],
    }

    summary_path = Path(output_config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_training_curves(
        log_csv_path=output_config["log_csv_path"],
        output_path=output_config["curves_path"],
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/baseline_cnn.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
