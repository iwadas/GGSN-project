"""Ensemble of all 4 trained models: baseline, HPO, evolutionary NAS, DARTS.

Runs inference with each model on the CIFAR-10 test set, then combines
predictions via soft voting (average logits) and hard voting (majority class).
Saves accuracy comparison and a bar chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.dataloader import get_cifar10_dataloaders
from evaluation.metrics import count_parameters
from evaluation.latency import measure_inference_latency
from models.baseline_cnn import build_baseline_cnn
from models.search_cnn import build_search_cnn_from_genome
from utils.config import load_yaml_config
from utils.reproducibility import set_seed


MODEL_KEYS = ["baseline", "hpo", "evolutionary", "darts"]

MODEL_LABELS = {
    "baseline": "Baseline CNN",
    "hpo": "HPO (Optuna)",
    "evolutionary": "Evolutionary NAS",
    "darts": "DARTS",
}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> np.ndarray:
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(targets.numpy(), predictions.numpy()):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved to {output_path}")

    per_class = np.diag(cm) / cm.sum(axis=1)
    print("  Per-class accuracy:")
    for idx, (name, acc) in enumerate(zip(CIFAR10_CLASSES, per_class)):
        print(f"    {name:12s}  {acc*100:.1f}%")

    return cm


def collect_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run full inference and return (all_logits, all_targets)."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def soft_voting(logits_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(logits_list).mean(dim=0)


def hard_voting(logits_list: list[torch.Tensor]) -> torch.Tensor:
    preds = torch.stack([l.argmax(dim=1) for l in logits_list], dim=0)
    mode, _ = torch.mode(preds, dim=0)
    return mode


def build_model(
    key: str,
    model_cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """Build model from config, load checkpoint, return (model, parameter_count)."""
    ckpt_path = Path(model_cfg["checkpoint_path"])

    if key == "baseline":
        yaml_cfg = load_yaml_config(model_cfg["config_path"])
        m = yaml_cfg["model"]
        model = build_baseline_cnn(
            num_layers=int(m["num_layers"]),
            base_filters=int(m["base_filters"]),
            filter_multiplier=int(m["filter_multiplier"]),
            kernel_size=int(m["kernel_size"]),
            dropout=float(m["dropout"]),
            num_classes=int(m.get("num_classes", 10)),
        )
    elif key == "hpo":
        summary = json.loads(Path(model_cfg["summary_path"]).read_text())
        bp = summary["best_params"]
        model = build_baseline_cnn(
            num_layers=int(bp["num_layers"]),
            base_filters=int(bp["base_filters"]),
            filter_multiplier=2,
            kernel_size=3,
            dropout=float(bp["dropout"]),
            num_classes=10,
        )
    else:
        summary = json.loads(Path(model_cfg["summary_path"]).read_text())
        genome = summary["best_model"]["genome"]
        model = build_search_cnn_from_genome(genome, num_classes=10)

    params = count_parameters(model)
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, params


def plot_comparison(
    results: dict[str, float],
    output_path: str | Path,
) -> None:
    names = list(results.keys())
    values = [results[n] * 100 for n in names]

    colors = ["#4A90D9", "#7B68EE", "#2E8B57", "#CD5C5C", "#FF8C00", "#8B4513"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="gray", linewidth=0.5)
    best_idx = int(np.argmax(values))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Ensemble Comparison — CIFAR-10", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved to {output_path}")


def run_ensemble(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_cfg = config["data"]
    _, _, test_loader = get_cifar10_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=int(data_cfg["batch_size"]),
        validation_ratio=float(data_cfg["validation_ratio"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        seed=seed,
        download=bool(data_cfg.get("download", True)),
    )

    models_cfg = config.get("models", {})
    output_cfg = config.get("outputs", {})
    summary_path = Path(output_cfg.get("summary_path", "results/ensemble_summary.json"))
    curves_path = Path(output_cfg.get("curves_path", "plots/ensemble_comparison.png"))
    cm_path = Path(output_cfg.get("confusion_matrix_path", "plots/ensemble_confusion_matrix.png"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    all_logits: list[torch.Tensor] = []
    targets: torch.Tensor | None = None
    individual_results: dict[str, dict] = {}

    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        mc = models_cfg.get(key)
        if mc is None:
            print(f"\n[SKIP] {label} — no config section")
            continue

        ckpt_path = Path(mc["checkpoint_path"])
        print(f"\n{'='*50}")
        print(f"Loading: {label}")
        print(f"Checkpoint: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"  [SKIP] File not found")
            continue

        model, params = build_model(key, mc, device)
        print(f"  Parameters: {params:,} ({params/1e3:.0f}K)")

        logits_batch, targets_batch = collect_logits(model, test_loader, device)
        acc = accuracy_from_logits(logits_batch, targets_batch)
        latency = measure_inference_latency(model, device=device)
        print(f"  Test accuracy: {acc*100:.2f}%")
        print(f"  Latency: {latency:.3f} ms")

        preds = logits_batch.argmax(dim=1)
        cm_path_key = Path(output_cfg.get(
            "confusion_matrix_path",
            "plots/ensemble_confusion_matrix.png",
        ))
        cm_individual_path = cm_path_key.parent / f"confusion_matrix_{key}.png"
        plot_confusion_matrix(
            targets_batch, preds, cm_individual_path,
            title=f"Confusion Matrix — {label}",
        )

        all_logits.append(logits_batch)
        if targets is None:
            targets = targets_batch

        individual_results[label] = {
            "test_accuracy": acc,
            "parameters": params,
            "latency_ms": latency,
            "checkpoint": str(ckpt_path),
        }

    if len(all_logits) < 2:
        raise RuntimeError(
            f"Need at least 2 models for ensemble, got {len(all_logits)}."
        )

    print(f"\n{'='*50}")
    print("Ensemble Results")
    print("="*50)

    soft_logits = soft_voting(all_logits)
    soft_acc = accuracy_from_logits(soft_logits, targets)
    print(f"  Soft voting (avg logits): {soft_acc*100:.2f}%")
    soft_preds = soft_logits.argmax(dim=1)

    hard_preds = hard_voting(all_logits)
    hard_acc = (hard_preds == targets).float().mean().item()
    print(f"  Hard voting (majority):   {hard_acc*100:.2f}%")

    print()
    cm = plot_confusion_matrix(
        targets, soft_preds, cm_path,
        title="Confusion Matrix — Ensemble (soft voting)",
    )

    ensemble_results: dict[str, float] = {}
    for name, res in individual_results.items():
        ensemble_results[name] = res["test_accuracy"]
    ensemble_results["Ensemble (soft)"] = soft_acc
    ensemble_results["Ensemble (hard)"] = hard_acc

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_models": len(all_logits),
        "models_included": list(individual_results.keys()),
        "individual_results": individual_results,
        "ensemble_soft_voting_accuracy": soft_acc,
        "ensemble_hard_voting_accuracy": hard_acc,
        "best_individual": max(
            individual_results.items(),
            key=lambda kv: kv[1]["test_accuracy"],
        )[0],
        "ensemble_improvement_over_best": soft_acc
        - max(r["test_accuracy"] for r in individual_results.values()),
        "confusion_matrix": cm.tolist(),
        "class_names": CIFAR10_CLASSES,
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] Summary saved to {summary_path}")

    plot_comparison(ensemble_results, curves_path)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/ensemble_config.yaml",
        help="Path to the YAML ensemble config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    summary = run_ensemble(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
