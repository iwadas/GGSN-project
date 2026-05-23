"""Fitness functions for neural architecture search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FitnessResult:
    """Fitness value with the metrics used to compute it."""

    fitness: float
    parameter_penalty: float = 0.0
    latency_penalty: float = 0.0


def accuracy_fitness(accuracy: float) -> FitnessResult:
    """Use validation accuracy directly as fitness."""
    _validate_accuracy(accuracy)
    return FitnessResult(fitness=accuracy)


def hardware_aware_fitness(
    accuracy: float,
    parameters: int,
    latency_ms: float,
    alpha: float,
    beta: float,
    parameter_scale: float = 1_000_000.0,
    latency_scale: float = 1.0,
) -> FitnessResult:
    """Compute Fitness = Accuracy - alpha * Params - beta * Latency.

    Params and latency are divided by scales before applying penalties. This
    keeps alpha and beta readable while preserving the TODO formula.
    """
    _validate_accuracy(accuracy)
    if parameters < 0:
        raise ValueError("parameters cannot be negative.")
    if latency_ms < 0.0:
        raise ValueError("latency_ms cannot be negative.")
    if parameter_scale <= 0.0:
        raise ValueError("parameter_scale must be positive.")
    if latency_scale <= 0.0:
        raise ValueError("latency_scale must be positive.")

    parameter_penalty = alpha * (parameters / parameter_scale)
    latency_penalty = beta * (latency_ms / latency_scale)
    return FitnessResult(
        fitness=accuracy - parameter_penalty - latency_penalty,
        parameter_penalty=parameter_penalty,
        latency_penalty=latency_penalty,
    )


def compute_fitness(
    accuracy: float,
    parameters: int,
    latency_ms: float | None,
    search_config: dict[str, Any],
    hardware_config: dict[str, Any] | None = None,
) -> FitnessResult:
    """Compute fitness using the configured search objective."""
    mode = str(search_config.get("fitness_mode", "accuracy")).lower()
    hardware_config = hardware_config or {}

    if mode == "accuracy":
        return accuracy_fitness(accuracy)
    if mode == "hardware_aware":
        if latency_ms is None:
            raise ValueError("latency_ms is required for hardware_aware fitness.")
        return hardware_aware_fitness(
            accuracy=accuracy,
            parameters=parameters,
            latency_ms=latency_ms,
            alpha=float(hardware_config.get("alpha", 0.0)),
            beta=float(hardware_config.get("beta", 0.0)),
            parameter_scale=float(hardware_config.get("parameter_scale", 1_000_000.0)),
            latency_scale=float(hardware_config.get("latency_scale", 1.0)),
        )

    raise ValueError(
        f"Unsupported fitness_mode: {mode}. Use 'accuracy' or 'hardware_aware'."
    )


def _validate_accuracy(accuracy: float) -> None:
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in the range [0, 1].")
