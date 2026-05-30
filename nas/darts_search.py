"""DARTS-inspired differentiable architecture search."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from data.dataloader import get_cifar10_dataloaders
from evaluation.latency import measure_inference_latency
from evaluation.metrics import count_parameters, evaluate
from hpo.optuna_search import build_optimizer
from models.darts_model import DartsCNN, OPS_NAMES, derive_architecture
from models.search_cnn import build_search_cnn_from_genome
from training.trainer import fit, get_default_device
from utils.plotting import plot_training_curves
from utils.reproducibility import set_seed


def search_architecture(
    model: DartsCNN,
    train_loader,
    validation_loader,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Train network weights and architecture parameters with alternating gradient descent.

    Returns a list of per-epoch alpha log dictionaries.
    """
    search_config = config["search"]
    training_config = config["training"]

    network_optimizer = build_optimizer(
        model=model,
        optimizer_name=str(training_config["optimizer"]),
        learning_rate=float(search_config.get("network_lr", training_config["learning_rate"])),
        weight_decay=float(training_config["weight_decay"]),
    )

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=float(search_config.get("arch_lr", 3e-4)),
        weight_decay=float(search_config.get("arch_weight_decay", 1e-3)),
    )

    criterion = nn.CrossEntropyLoss()
    search_epochs = int(search_config.get("search_epochs", 5))
    initial_temp = float(search_config.get("temperature", 1.0))
    final_temp = float(search_config.get("temperature_final", 0.1))
    arch_entropy_weight = float(search_config.get("arch_entropy_weight", 0.0))
    scaler_enabled = bool(training_config["use_mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)

    alpha_log: list[dict[str, Any]] = []

    for epoch in range(1, search_epochs + 1):
        if search_epochs > 1:
            model.temperature = initial_temp + (final_temp - initial_temp) * (epoch - 1) / (search_epochs - 1)
        else:
            model.temperature = initial_temp

        model.train()

        for inputs, targets in tqdm(train_loader, desc=f"train w epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            network_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(network_optimizer)
                scaler.update()
            else:
                loss.backward()
                network_optimizer.step()

        model.train()
        for inputs, targets in tqdm(validation_loader, desc=f"train a epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            arch_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
                if arch_entropy_weight > 0.0:
                    entropy = sum(
                        -(torch.softmax(alpha / model.temperature, dim=0)
                          * torch.log_softmax(alpha / model.temperature, dim=0)).sum()
                        for alpha in model.arch_parameters()
                    )
                    loss = loss + arch_entropy_weight * entropy

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(arch_optimizer)
                scaler.update()
            else:
                loss.backward()
                arch_optimizer.step()

        epoch_alphas: dict[str, Any] = {
            "epoch": epoch,
            "temperature": model.temperature,
            "train_loss": float(loss.item()),
        }
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.layers):
                weights = torch.softmax(layer.mixed_op.alpha, dim=0)
                epoch_alphas[f"layer{layer_idx}_selected"] = OPS_NAMES[int(weights.argmax().item())]
                for op_idx, op_name in enumerate(OPS_NAMES):
                    epoch_alphas[f"layer{layer_idx}_{op_name}"] = float(weights[op_idx].item())
        alpha_log.append(epoch_alphas)

        tqdm.write(f"search epoch {epoch}/{search_epochs}  temp={model.temperature:.3f}")

    return alpha_log


def save_alpha_log(alpha_log: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save per-epoch alpha weights to CSV."""
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        if not alpha_log:
            return
        writer = csv.DictWriter(file, fieldnames=list(alpha_log[0].keys()))
        writer.writeheader()
        writer.writerows(alpha_log)


def plot_alpha_convergence(
    alpha_log_path: str | Path,
    output_path: str | Path,
    num_layers: int,
) -> None:
    """Plot alpha weight evolution for each layer across search epochs."""
    import matplotlib.pyplot as plt

    alpha_df = pd.read_csv(alpha_log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=num_layers, figsize=(9, 3 * num_layers), squeeze=False)
    for layer_idx in range(num_layers):
        ax = axes[layer_idx][0]
        for op_name in OPS_NAMES:
            col = f"layer{layer_idx}_{op_name}"
            if col in alpha_df.columns:
                ax.plot(alpha_df["epoch"], alpha_df[col], marker="o", label=op_name)
        ax.set_ylabel(f"Layer {layer_idx} weight")
        ax.set_title(f"Layer {layer_idx} — selected: {alpha_df[f'layer{layer_idx}_selected'].iloc[-1]}")
        ax.legend(fontsize="small")
    axes[-1][0].set_xlabel("Search epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_derived_architecture(
    genome: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build, train, and evaluate a discrete SearchCNN from the derived genome."""
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

    model = build_search_cnn_from_genome(
        genome,
        num_classes=int(model_config["num_classes"]),
    )
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=str(training_config["optimizer"]),
        learning_rate=float(training_config["learning_rate"]),
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
    hardware_config = config.get("hardware_aware", {})
    latency_ms = measure_inference_latency(
        model,
        input_shape=tuple(hardware_config.get("input_shape", (1, 3, 32, 32))),
        device=device,
        warmup_steps=int(hardware_config.get("latency_warmup_steps", 20)),
        measured_steps=int(hardware_config.get("latency_measured_steps", 100)),
    )
    plot_training_curves(
        log_csv_path=output_config["best_log_csv_path"],
        output_path=output_config["best_curves_path"],
    )

    return {
        "genome": genome,
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


def create_comparison_with_evolutionary_nas(
    darts_summary: dict[str, Any],
    evo_summary_path: str | Path,
) -> dict[str, Any] | None:
    """Compare DARTS results with evolutionary NAS results if available."""
    evo_path = Path(evo_summary_path)
    if not evo_path.exists():
        return None

    with evo_path.open("r", encoding="utf-8") as file:
        evo_summary = json.load(file)

    evo_best = evo_summary.get("best_model", {})
    darts_best = darts_summary.get("best_model", {})

    return {
        "evolutionary_nas_summary_path": str(evo_path),
        "darts_test_accuracy": darts_best.get("test_accuracy"),
        "evolutionary_test_accuracy": evo_best.get("test_accuracy"),
        "test_accuracy_delta": (
            darts_best.get("test_accuracy", 0.0) - evo_best.get("test_accuracy", 0.0)
            if darts_best.get("test_accuracy") is not None
            and evo_best.get("test_accuracy") is not None
            else None
        ),
        "darts_parameters": darts_best.get("parameters"),
        "evolutionary_parameters": evo_best.get("parameters"),
        "darts_latency_ms": darts_best.get("latency_ms"),
        "evolutionary_latency_ms": evo_best.get("latency_ms"),
    }


def run_darts_search(config: dict[str, Any]) -> dict[str, Any]:
    """Run DARTS-inspired differentiable architecture search.

    Phases:
      1. Differentiable search — train network weights and architecture alphas.
      2. Derive discrete architecture from learned alphas.
      3. Retrain discrete architecture from scratch and evaluate on test set.
      4. Compare with evolutionary NAS results (if available).
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_config = config["outputs"]
    data_config = config["data"]
    model_config = config["model"]
    search_config = config["search"]
    device = get_default_device()

    train_loader, validation_loader, _ = get_cifar10_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=int(data_config["batch_size"]),
        validation_ratio=float(data_config["validation_ratio"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        download=bool(data_config["download"]),
    )

    filters = [int(f) for f in search_config.get("filters", model_config.get("filters", [32, 64, 128]))]
    dropout = float(search_config.get("dropout", model_config.get("dropout", 0.0)))

    model = DartsCNN(
        filters=filters,
        dropout=dropout,
        num_classes=int(model_config["num_classes"]),
    ).to(device)

    tqdm.write(f"DARTS search model: {sum(p.numel() for p in model.network_parameters())} "
               f"network params, {sum(p.numel() for p in model.arch_parameters())} arch params")

    alpha_log = search_architecture(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
    )

    alpha_log_path = Path(output_config["alpha_log_csv_path"])
    alpha_log_path.parent.mkdir(parents=True, exist_ok=True)
    save_alpha_log(alpha_log, alpha_log_path)

    plot_alpha_convergence(
        alpha_log_path=alpha_log_path,
        output_path=output_config["alpha_convergence_plot_path"],
        num_layers=len(model.layers),
    )

    genome = derive_architecture(model, dropout=dropout)

    derived_genome_path = Path(output_config["derived_genome_path"])
    derived_genome_path.parent.mkdir(parents=True, exist_ok=True)
    with derived_genome_path.open("w", encoding="utf-8") as file:
        json.dump(genome, file, indent=2)

    best_model_summary = train_derived_architecture(genome, config)

    evo_comparison = create_comparison_with_evolutionary_nas(
        {"best_model": best_model_summary},
        output_config.get("evolutionary_summary_path", "results/evolutionary_summary.json"),
    )

    final_arch_alphas = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            weights = torch.softmax(layer.mixed_op.alpha, dim=0)
            for op_idx, op_name in enumerate(OPS_NAMES):
                final_arch_alphas[f"layer{layer_idx}_{op_name}"] = float(weights[op_idx].item())

    summary = {
        "search_epochs": int(search_config.get("search_epochs", 5)),
        "network_parameters": sum(p.numel() for p in model.network_parameters()),
        "arch_parameters": sum(p.numel() for p in model.arch_parameters()),
        "final_architecture_weights": final_arch_alphas,
        "derived_genome": genome,
        "best_model": best_model_summary,
    }
    if evo_comparison is not None:
        summary["comparison_with_evolutionary_nas"] = evo_comparison

    summary_path = Path(output_config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
