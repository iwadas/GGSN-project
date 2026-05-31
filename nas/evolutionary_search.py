"""Evolutionary architecture search for CIFAR-10 CNNs."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataloader import get_cifar10_dataloaders
from evaluation.latency import measure_inference_latency
from evaluation.metrics import count_parameters, evaluate
from evaluation.pareto import plot_pareto_analysis
from hpo.optuna_search import build_optimizer
from models.search_cnn import build_search_cnn_from_genome
from nas.fitness import compute_fitness
from nas.mutation import Genome, crossover_genomes, mutate_genome, random_genome
from nas.selection import tournament_selection
from training.trainer import fit, get_default_device, train_one_epoch
from utils.plotting import plot_training_curves
from utils.reproducibility import set_seed


@dataclass(frozen=True)
class EvaluatedIndividual:
    generation: int
    individual_id: int
    genome: Genome
    fitness: float
    validation_accuracy: float
    train_accuracy: float
    train_loss: float
    validation_loss: float
    parameters: int
    latency_ms: float | None
    parameter_penalty: float
    latency_penalty: float

    def as_record(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "individual_id": self.individual_id,
            "fitness": self.fitness,
            "validation_accuracy": self.validation_accuracy,
            "train_accuracy": self.train_accuracy,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "parameters": self.parameters,
            "latency_ms": self.latency_ms,
            "parameter_penalty": self.parameter_penalty,
            "latency_penalty": self.latency_penalty,
            "genome": json.dumps(self.genome),
            "num_layers": self.genome["num_layers"],
            "filters": json.dumps(self.genome["filters"]),
            "kernel_sizes": json.dumps(self.genome["kernel_sizes"]),
            "pooling_types": json.dumps(self.genome["pooling_types"]),
            "skip_connections": json.dumps(self.genome["skip_connections"]),
            "dropout": self.genome["dropout"],
        }


def append_records_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append dictionaries to a CSV file."""
    if not records:
        return

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        if should_write_header:
            writer.writeheader()
        writer.writerows(records)


def evaluate_genome(
    genome: Genome,
    individual_id: int,
    generation: int,
    config: dict[str, Any],
    train_loader,
    validation_loader,
) -> EvaluatedIndividual:
    """Train one candidate briefly and return its validation fitness."""
    training_config = config["training"]
    search_config = config["search"]
    model_config = config["model"]
    hardware_config = config.get("hardware_aware", {})
    device = get_default_device()

    model = build_search_cnn_from_genome(
        genome,
        num_classes=int(model_config["num_classes"]),
    ).to(device)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=str(training_config["optimizer"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=bool(training_config["use_mixed_precision"]) and device.type == "cuda",
    )

    train_result = None
    validation_result = None
    for _ in range(int(search_config["candidate_epochs"])):
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_mixed_precision=bool(training_config["use_mixed_precision"]),
        )
        validation_result = evaluate(model, validation_loader, criterion, device)

    if train_result is None or validation_result is None:
        raise ValueError("candidate_epochs must be at least 1.")

    validation_accuracy = validation_result.accuracy
    parameters = count_parameters(model)
    latency_ms = None
    if str(search_config.get("fitness_mode", "accuracy")).lower() == "hardware_aware":
        latency_ms = measure_inference_latency(
            model=model,
            input_shape=tuple(hardware_config.get("input_shape", (1, 3, 32, 32))),
            device=device,
            warmup_steps=int(hardware_config.get("latency_warmup_steps", 10)),
            measured_steps=int(hardware_config.get("latency_measured_steps", 30)),
        )
    fitness_result = compute_fitness(
        accuracy=validation_accuracy,
        parameters=parameters,
        latency_ms=latency_ms,
        search_config=search_config,
        hardware_config=hardware_config,
    )

    return EvaluatedIndividual(
        generation=generation,
        individual_id=individual_id,
        genome=genome,
        fitness=fitness_result.fitness,
        validation_accuracy=validation_accuracy,
        train_accuracy=train_result.accuracy,
        train_loss=train_result.loss,
        validation_loss=validation_result.loss,
        parameters=parameters,
        latency_ms=latency_ms,
        parameter_penalty=fitness_result.parameter_penalty,
        latency_penalty=fitness_result.latency_penalty,
    )


def create_initial_population(
    config: dict[str, Any],
    rng: random.Random,
) -> list[Genome]:
    """Create the initial population of genomes."""
    return [
        random_genome(config["search_space"], rng)
        for _ in range(int(config["search"]["population_size"]))
    ]


def plot_evolution_history(records_path: str | Path, output_path: str | Path) -> None:
    """Plot best and mean fitness per generation."""
    import matplotlib.pyplot as plt

    history = pd.read_csv(records_path)
    grouped = history.groupby("generation")["fitness"]
    best = grouped.max()
    mean = grouped.mean()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(best.index, best.values, marker="o", label="best fitness")
    ax.plot(mean.index, mean.values, marker="o", label="mean fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolutionary NAS Progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_best_architecture(
    best_genome: Genome,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Retrain the best architecture and evaluate it on the test set."""
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
        best_genome,
        num_classes=int(model_config["num_classes"]),
    )
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=str(training_config["optimizer"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["final_epochs"]),
    )
    criterion = nn.CrossEntropyLoss()

    training_result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
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
        "genome": best_genome,
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


def run_evolutionary_search(config: dict[str, Any]) -> dict[str, Any]:
    """Run evolutionary NAS and return a summary dictionary."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    rng = random.Random(seed)

    output_config = config["outputs"]
    records_path = Path(output_config["population_csv_path"])
    if records_path.exists():
        records_path.unlink()

    data_config = config["data"]
    train_loader, validation_loader, _ = get_cifar10_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=int(data_config["batch_size"]),
        validation_ratio=float(data_config["validation_ratio"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        download=bool(data_config["download"]),
    )

    initial_population = create_initial_population(config, rng)
    active_population: list[dict[str, Any]] = []
    best_individual: dict[str, Any] | None = None
    individual_id = 0
    age_counter = 0

    initial_records = []
    for genome in initial_population:
        evaluated = evaluate_genome(
            genome=genome,
            individual_id=individual_id,
            generation=0,
            config=config,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        individual_id += 1
        record = evaluated.as_record()
        active_individual = {
            "genome": evaluated.genome,
            "fitness": evaluated.fitness,
            "record": record,
            "age": age_counter,
        }
        age_counter += 1
        active_population.append(active_individual)
        initial_records.append(record)

        if best_individual is None or evaluated.fitness > best_individual["fitness"]:
            best_individual = active_individual

    append_records_csv(records_path, initial_records)

    children_per_generation = int(config["search"]["children_per_generation"])
    crossover_rate = float(config["search"].get("crossover_rate", 0.0))
    for generation in range(1, int(config["search"]["generations"]) + 1):
        generation_records = []
        for _ in range(children_per_generation):
            parent = tournament_selection(
                active_population,
                int(config["search"]["tournament_size"]),
                rng,
            )

            if crossover_rate > 0.0 and rng.random() < crossover_rate and len(active_population) >= 2:
                parent_b = tournament_selection(
                    active_population,
                    int(config["search"]["tournament_size"]),
                    rng,
                )
                child_genome = crossover_genomes(
                    parent["genome"],
                    parent_b["genome"],
                    search_space=config["search_space"],
                    rng=rng,
                )
            else:
                child_genome = parent["genome"]

            child_genome = mutate_genome(
                child_genome,
                search_space=config["search_space"],
                rng=rng,
                mutation_rate=float(config["search"]["mutation_rate"]),
            )

            evaluated = evaluate_genome(
                genome=child_genome,
                individual_id=individual_id,
                generation=generation,
                config=config,
                train_loader=train_loader,
                validation_loader=validation_loader,
            )
            individual_id += 1
            record = evaluated.as_record()
            active_individual = {
                "genome": evaluated.genome,
                "fitness": evaluated.fitness,
                "record": record,
                "age": age_counter,
            }
            age_counter += 1
            active_population.append(active_individual)
            generation_records.append(record)

            if best_individual is None or evaluated.fitness > best_individual["fitness"]:
                best_individual = active_individual

        population_size = int(config["search"]["population_size"])
        if len(active_population) > population_size:
            active_population.sort(key=lambda x: x["fitness"], reverse=True)
            active_population = active_population[:population_size]

        append_records_csv(records_path, generation_records)

    if best_individual is None:
        raise ValueError("Evolutionary search produced no evaluated individuals.")

    best_genome_path = Path(output_config["best_genome_path"])
    best_genome_path.parent.mkdir(parents=True, exist_ok=True)
    with best_genome_path.open("w", encoding="utf-8") as file:
        json.dump(best_individual["genome"], file, indent=2)

    plot_evolution_history(
        records_path=records_path,
        output_path=output_config["evolution_plot_path"],
    )
    pareto_summary = create_pareto_analysis(records_path, output_config)
    best_model_summary = train_best_architecture(best_individual["genome"], config)

    summary = {
        "generations": int(config["search"]["generations"]),
        "population_size": int(config["search"]["population_size"]),
        "children_per_generation": children_per_generation,
        "evaluated_individuals": individual_id,
        "best_search_fitness": best_individual["fitness"],
        "best_search_record": best_individual["record"],
        "pareto": pareto_summary,
        "best_model": best_model_summary,
    }

    summary_path = Path(output_config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def create_pareto_analysis(
    records_path: str | Path,
    output_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute Pareto-efficient candidates and save trade-off plots."""
    pareto_csv_path = output_config.get(
        "pareto_csv_path",
        "results/evolutionary_pareto_frontier.csv",
    )
    accuracy_latency_plot_path = output_config.get(
        "accuracy_latency_plot_path",
        "plots/evolutionary_accuracy_vs_latency.png",
    )
    accuracy_parameters_plot_path = output_config.get(
        "accuracy_parameters_plot_path",
        "plots/evolutionary_accuracy_vs_parameters.png",
    )
    pareto_plot_path = output_config.get(
        "pareto_plot_path",
        "plots/evolutionary_pareto_frontier.png",
    )

    pareto = plot_pareto_analysis(
        results_csv_path=records_path,
        pareto_csv_path=pareto_csv_path,
        accuracy_latency_path=accuracy_latency_plot_path,
        accuracy_parameters_path=accuracy_parameters_plot_path,
        pareto_frontier_path=pareto_plot_path,
    )

    return {
        "pareto_count": int(len(pareto)),
        "pareto_csv_path": str(pareto_csv_path),
        "accuracy_latency_plot_path": str(accuracy_latency_plot_path),
        "accuracy_parameters_plot_path": str(accuracy_parameters_plot_path),
        "pareto_plot_path": str(pareto_plot_path),
    }
