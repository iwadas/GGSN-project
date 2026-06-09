"""Knowledge Distillation — compress HPO teacher into DARTS student.

Grid search over temperature (T) and alpha (α) to recover the accuracy gap.

Usage:
    python scripts/run_distillation.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataloader import get_cifar10_dataloaders
from evaluation.metrics import evaluate, count_parameters
from models.baseline_cnn import build_baseline_cnn
from models.search_cnn import build_search_cnn_from_genome
from training.distillation import distill_epoch, precompute_teacher_logits
from utils.reproducibility import set_seed


TEMPERATURES = [1, 2, 4, 8]
ALPHAS = [0.3, 0.5, 0.7]

TEACHER_CKPT = Path("checkpoints/hpo_best_baseline_cnn.pt")
STUDENT_GENOME_PATH = Path("results/darts_derived_genome.json")
STUDENT_CKPT = Path("checkpoints/darts_best_cnn.pt")

STUDENT_BASELINE_ACC = 0.806  # from darts_summary.json — accuracy without KD

OUTPUT_DIR = Path("knowledge_distillation")
PLOTS_DIR = Path("plots")

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
SEED = 42
NUM_WORKERS = 0


def build_teacher() -> nn.Module:
    summary_path = PROJECT_ROOT / "results/hpo_summary.json"
    summary = json.loads(summary_path.read_text())
    bp = summary["best_params"]
    model = build_baseline_cnn(
        num_layers=int(bp["num_layers"]),
        base_filters=int(bp["base_filters"]),
        filter_multiplier=2,
        kernel_size=3,
        dropout=float(bp["dropout"]),
        num_classes=10,
    )
    ckpt = torch.load(TEACHER_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_pretrained_student() -> nn.Module:
    genome = json.loads(STUDENT_GENOME_PATH.read_text())
    model = build_search_cnn_from_genome(genome, num_classes=10)
    ckpt = torch.load(STUDENT_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def run_distillation() -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=str(PROJECT_ROOT / "data/raw"),
        batch_size=BATCH_SIZE,
        validation_ratio=0.1,
        num_workers=NUM_WORKERS,
        seed=SEED,
        download=True,
    )

    print("\nBuilding teacher (HPO)...")
    teacher = build_teacher()
    teacher_params = count_parameters(teacher)
    teacher.to(device)
    teacher.eval()
    print(f"  Teacher parameters: {teacher_params:,} ({teacher_params/1e3:.0f}K)")

    teacher_test_loss, teacher_test_acc = evaluate(
        teacher, test_loader, nn.CrossEntropyLoss(), device
    )
    print(f"  Teacher test accuracy: {teacher_test_acc*100:.2f}%")

    student_params = count_parameters(load_pretrained_student())
    print(f"\nStudent architecture (DARTS): {student_params:,} ({student_params/1e3:.0f}K)")
    compression_ratio = teacher_params / student_params
    print(f"  Compression ratio: {compression_ratio:.1f}×")
    print(f"  Student baseline accuracy (no KD): {STUDENT_BASELINE_ACC*100:.2f}%")

    print("\nPrecomputing teacher logits on training set...")
    teacher_logits = precompute_teacher_logits(teacher, train_loader, device)
    teacher.to("cpu")
    print(f"  Teacher logits shape: {teacher_logits.shape}")

    results: list[dict] = []

    for temperature, alpha in itertools.product(TEMPERATURES, ALPHAS):
        print(f"\n{'='*60}")
        print(f"Distillation T={temperature}, α={alpha}")
        print(f"{'='*60}")

        student = load_pretrained_student()
        student.to(device)
        optimizer = torch.optim.Adam(
            student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        start_time = time.perf_counter()
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = distill_epoch(
                student=student,
                teacher_logits=teacher_logits,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                temperature=temperature,
                alpha=alpha,
            )
            val_result = evaluate(student, val_loader, nn.CrossEntropyLoss(), device)

            if val_result.accuracy > best_val_acc:
                best_val_acc = val_result.accuracy
                best_epoch = epoch

            print(
                f"  epoch {epoch:2d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_acc={val_result.accuracy:.4f}"
            )

        elapsed = time.perf_counter() - start_time

        test_result = evaluate(student, test_loader, nn.CrossEntropyLoss(), device)
        accuracy_gap = teacher_test_acc - test_result.accuracy
        gap_to_recover = teacher_test_acc - STUDENT_BASELINE_ACC
        recovered = test_result.accuracy - STUDENT_BASELINE_ACC
        recovery = max(0.0, recovered / gap_to_recover) if gap_to_recover > 0 else 0.0

        entry = {
            "temperature": temperature,
            "alpha": alpha,
            "epochs": EPOCHS,
            "best_validation_accuracy": best_val_acc,
            "best_validation_epoch": best_epoch,
            "test_loss": test_result.loss,
            "test_accuracy": test_result.accuracy,
            "train_loss_final": train_loss,
            "train_accuracy_final": train_acc,
            "teacher_test_accuracy": teacher_test_acc,
            "accuracy_gap": accuracy_gap,
            "recovery_ratio": recovery,
            "training_time_seconds": elapsed,
        }
        results.append(entry)

        print(f"  Test accuracy: {test_result.accuracy*100:.2f}%")
        print(f"  Gap to teacher: {accuracy_gap*100:.2f} pp")
        print(f"  Recovery: {recovery*100:.1f}%")

        del student

    results.sort(key=lambda r: r["test_accuracy"], reverse=True)
    best = results[0]

    summary = {
        "teacher": {
            "type": "HPO (BaselineCNN)",
            "parameters": teacher_params,
            "test_accuracy": teacher_test_acc,
            "checkpoint": str(TEACHER_CKPT),
        },
        "student": {
            "type": "DARTS-derived SearchCNN",
            "parameters": student_params,
            "baseline_test_accuracy": STUDENT_BASELINE_ACC,
            "checkpoint": str(STUDENT_CKPT),
            "genome": json.loads(STUDENT_GENOME_PATH.read_text()),
            "compression_ratio": compression_ratio,
        },
        "training": {
            "epochs_per_variant": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "number_of_variants": len(results),
            "total_variants_trained": len(results),
        },
        "grid_search_results": results,
        "best_variant": best,
    }

    summary_path = OUTPUT_DIR / "distillation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] Summary saved to {summary_path}")

    plot_results(results, OUTPUT_DIR / "distillation_comparison.png")

    return summary


def plot_results(results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "test_accuracy", "Test Accuracy (%)", "Test Accuracy by T and α"),
        (axes[1], "recovery_ratio", "Recovery Ratio", "Recovery Ratio by T and α"),
    ]:
        for alpha_group in ALPHAS:
            subset = [r for r in results if r["alpha"] == alpha_group]
            temps = [r["temperature"] for r in subset]
            values = [r[metric] * 100 for r in subset]
            ax.plot(temps, values, marker="o", label=f"α={alpha_group}")
        ax.set_xlabel("Temperature (T)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved to {output_path}")


if __name__ == "__main__":
    summary = run_distillation()
    print(json.dumps(summary, indent=2))
