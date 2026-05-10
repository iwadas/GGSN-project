"""Optuna search for the configurable baseline CNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch
from torch import nn

from data.dataloader import get_cifar10_dataloaders
from evaluation.metrics import count_parameters, evaluate, measure_inference_latency
from models.baseline_cnn import build_baseline_cnn
from training.trainer import fit, get_default_device, train_one_epoch
from utils.plotting import plot_training_curves
from utils.reproducibility import set_seed


def suggest_hyperparameters(
    trial: optuna.Trial,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Sample one HPO configuration from the YAML-defined search space."""
    search_config = config["search"]

    learning_rate_config = search_config["learning_rate"]
    dropout_config = search_config["dropout"]
    num_layers_config = search_config["num_layers"]

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(learning_rate_config["low"]),
            float(learning_rate_config["high"]),
            log=bool(learning_rate_config.get("log", False)),
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            list(search_config["batch_size"]["choices"]),
        ),
        "optimizer": trial.suggest_categorical(
            "optimizer",
            list(search_config["optimizer"]["choices"]),
        ),
        "dropout": trial.suggest_float(
            "dropout",
            float(dropout_config["low"]),
            float(dropout_config["high"]),
        ),
        "base_filters": trial.suggest_categorical(
            "base_filters",
            list(search_config["base_filters"]["choices"]),
        ),
        "num_layers": trial.suggest_int(
            "num_layers",
            int(num_layers_config["low"]),
            int(num_layers_config["high"]),
        ),
    }


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create an optimizer from a simple name."""
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if normalized_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_pruner(config: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Create the configured Optuna pruner."""
    pruner_config = config["search"].get("pruner", {"type": "median"})
    pruner_type = pruner_config.get("type", "median")

    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_config.get("n_startup_trials", 3)),
            n_warmup_steps=int(pruner_config.get("n_warmup_steps", 1)),
        )
    if pruner_type == "none":
        return optuna.pruners.NopPruner()

    raise ValueError(f"Unsupported pruner type: {pruner_type}")


def create_objective(config: dict[str, Any]):
    """Create an Optuna objective function for validation accuracy."""
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    seed = int(config.get("seed", 42))

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        set_seed(trial_seed)
        params = suggest_hyperparameters(trial, config)

        train_loader, validation_loader, _ = get_cifar10_dataloaders(
            data_dir=data_config["data_dir"],
            batch_size=int(params["batch_size"]),
            validation_ratio=float(data_config["validation_ratio"]),
            num_workers=int(data_config["num_workers"]),
            seed=seed,
            download=bool(data_config["download"]),
        )

        device = get_default_device()
        model = build_baseline_cnn(
            num_layers=int(params["num_layers"]),
            base_filters=int(params["base_filters"]),
            filter_multiplier=int(model_config["filter_multiplier"]),
            kernel_size=int(model_config["kernel_size"]),
            dropout=float(params["dropout"]),
            num_classes=int(model_config["num_classes"]),
        ).to(device)

        optimizer = build_optimizer(
            model=model,
            optimizer_name=str(params["optimizer"]),
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=bool(training_config["use_mixed_precision"])
            and device.type == "cuda",
        )

        best_validation_accuracy = 0.0
        for epoch in range(1, int(training_config["trial_epochs"]) + 1):
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_mixed_precision=bool(training_config["use_mixed_precision"]),
            )
            validation_result = evaluate(model, validation_loader, criterion, device)
            best_validation_accuracy = max(
                best_validation_accuracy,
                validation_result.accuracy,
            )

            trial.report(validation_result.accuracy, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("parameters", count_parameters(model))
        return best_validation_accuracy

    return objective


def plot_optimization_history(study: optuna.Study, output_path: str | Path) -> None:
    """Save a simple Optuna optimization history plot."""
    import matplotlib.pyplot as plt

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        return

    trial_numbers = [trial.number for trial in completed_trials]
    values = [float(trial.value) for trial in completed_trials]
    best_values = []
    best_so_far = float("-inf")
    for value in values:
        best_so_far = max(best_so_far, value)
        best_values.append(best_so_far)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trial_numbers, values, marker="o", label="trial validation accuracy")
    ax.plot(trial_numbers, best_values, marker="o", label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("HPO Optimization History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_study_outputs(study: optuna.Study, config: dict[str, Any]) -> None:
    """Save trials, best parameters, and optimization plot."""
    output_config = config["outputs"]

    trials_path = Path(output_config["trials_csv_path"])
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params_path = Path(output_config["best_params_path"])
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with best_params_path.open("w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)

    plot_optimization_history(study, output_config["optimization_plot_path"])


def train_best_model(config: dict[str, Any], best_params: dict[str, Any]) -> dict[str, Any]:
    """Retrain the best HPO configuration and evaluate it on the test set."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]

    train_loader, validation_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=int(best_params["batch_size"]),
        validation_ratio=float(data_config["validation_ratio"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        download=bool(data_config["download"]),
    )

    model = build_baseline_cnn(
        num_layers=int(best_params["num_layers"]),
        base_filters=int(best_params["base_filters"]),
        filter_multiplier=int(model_config["filter_multiplier"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(best_params["dropout"]),
        num_classes=int(model_config["num_classes"]),
    )
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=str(best_params["optimizer"]),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    training_result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=int(training_config["final_epochs"]),
        use_mixed_precision=bool(training_config["use_mixed_precision"]),
        early_stopping_patience=int(training_config["early_stopping_patience"]),
        checkpoint_path=output_config["best_checkpoint_path"],
        log_csv_path=output_config["best_log_csv_path"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output_config["best_checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_result = evaluate(model, test_loader, criterion, device)
    latency_ms = measure_inference_latency(model, device=device)
    plot_training_curves(
        log_csv_path=output_config["best_log_csv_path"],
        output_path=output_config["best_curves_path"],
    )

    return {
        **best_params,
        "final_epochs": int(training_config["final_epochs"]),
        "parameters": parameter_count,
        "best_validation_accuracy": training_result.best_validation_accuracy,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "latency_ms": latency_ms,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": output_config["best_checkpoint_path"],
        "log_csv_path": output_config["best_log_csv_path"],
    }


def run_hpo(config: dict[str, Any]) -> dict[str, Any]:
    """Run Optuna search, save artifacts, retrain best model, and return summary."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    study = optuna.create_study(
        study_name=config["search"]["study_name"],
        direction=config["search"].get("direction", "maximize"),
        pruner=create_pruner(config),
    )
    timeout = config["search"].get("timeout_seconds")
    study.optimize(
        create_objective(config),
        n_trials=int(config["search"]["n_trials"]),
        timeout=None if timeout is None else int(timeout),
    )

    save_study_outputs(study, config)
    best_model_summary = train_best_model(config, study.best_params)

    summary = {
        "study_name": config["search"]["study_name"],
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_trial_validation_accuracy": study.best_value,
        "best_params": study.best_params,
        "best_model": best_model_summary,
    }

    baseline_summary_path = config["outputs"].get("baseline_summary_path")
    if baseline_summary_path and Path(baseline_summary_path).exists():
        with Path(baseline_summary_path).open("r", encoding="utf-8") as file:
            baseline_summary = json.load(file)
        summary["baseline_comparison"] = {
            "baseline_summary_path": baseline_summary_path,
            "baseline_test_accuracy": baseline_summary.get("test_accuracy"),
            "hpo_test_accuracy": best_model_summary["test_accuracy"],
            "test_accuracy_delta": best_model_summary["test_accuracy"]
            - baseline_summary.get("test_accuracy", 0.0),
            "baseline_parameters": baseline_summary.get("parameters"),
            "hpo_parameters": best_model_summary["parameters"],
            "parameter_delta": best_model_summary["parameters"]
            - baseline_summary.get("parameters", 0),
            "baseline_latency_ms": baseline_summary.get("latency_ms"),
            "hpo_latency_ms": best_model_summary["latency_ms"],
            "latency_delta_ms": best_model_summary["latency_ms"]
            - baseline_summary.get("latency_ms", 0.0),
        }

    summary_path = Path(config["outputs"]["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
