"""Selection operators for evolutionary NAS."""

from __future__ import annotations

import random
from typing import Any


Individual = dict[str, Any]


def select_elite(population: list[Individual], elite_size: int) -> list[Individual]:
    """Return the best individuals by fitness."""
    return sorted(
        population,
        key=lambda individual: float(individual["fitness"]),
        reverse=True,
    )[:elite_size]


def tournament_selection(
    population: list[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    """Select one individual using tournament selection."""
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1.")
    if not population:
        raise ValueError("population cannot be empty.")

    candidates = rng.sample(population, k=min(tournament_size, len(population)))
    return max(candidates, key=lambda individual: float(individual["fitness"]))
