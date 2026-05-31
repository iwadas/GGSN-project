"""Genome creation and mutation for evolutionary NAS."""

from __future__ import annotations

import copy
import random
from typing import Any


Genome = dict[str, Any]


def _generate_monotonic_filters(
    search_space: dict[str, Any],
    num_layers: int,
    rng: random.Random,
) -> list[int]:
    """Generate non-decreasing filter counts for a more regular architecture."""
    choices = sorted(search_space["filters"]["choices"])
    filters: list[int] = []
    current_min_idx = 0
    for _ in range(num_layers):
        idx = rng.randint(current_min_idx, len(choices) - 1)
        filters.append(int(choices[idx]))
        current_min_idx = idx
    return filters


def random_genome(search_space: dict[str, Any], rng: random.Random) -> Genome:
    """Create one random architecture genome from a search-space config."""
    num_layers = rng.randint(
        int(search_space["num_layers"]["low"]),
        int(search_space["num_layers"]["high"]),
    )
    return {
        "num_layers": num_layers,
        "filters": _generate_monotonic_filters(search_space, num_layers, rng),
        "kernel_sizes": [
            int(rng.choice(search_space["kernel_sizes"]["choices"]))
            for _ in range(num_layers)
        ],
        "pooling_types": [
            str(rng.choice(search_space["pooling_types"]["choices"]))
            for _ in range(num_layers)
        ],
        "skip_connections": [
            bool(rng.choice(search_space["skip_connections"]["choices"]))
            for _ in range(num_layers)
        ],
        "dilations": [1] * num_layers,
        "separable": [False] * num_layers,
        "dropout": rng.uniform(
            float(search_space["dropout"]["low"]),
            float(search_space["dropout"]["high"]),
        ),
    }


def normalize_genome_layers(
    genome: Genome,
    search_space: dict[str, Any],
    rng: random.Random,
) -> Genome:
    """Ensure all per-layer fields match genome['num_layers']."""
    normalized = copy.deepcopy(genome)
    target_layers = int(normalized["num_layers"])

    layer_fields = {
        "filters": search_space["filters"]["choices"],
        "kernel_sizes": search_space["kernel_sizes"]["choices"],
        "pooling_types": search_space["pooling_types"]["choices"],
        "skip_connections": search_space["skip_connections"]["choices"],
    }

    for field, choices in layer_fields.items():
        values = list(normalized[field])
        while len(values) < target_layers:
            values.append(rng.choice(choices))
        normalized[field] = values[:target_layers]

    for field, default in [("dilations", 1), ("separable", False)]:
        if field not in normalized:
            normalized[field] = [default] * target_layers
        values = list(normalized[field])
        while len(values) < target_layers:
            values.append(default)
        normalized[field] = values[:target_layers]

    normalized["filters"] = [int(value) for value in normalized["filters"]]
    normalized["kernel_sizes"] = [int(value) for value in normalized["kernel_sizes"]]
    normalized["pooling_types"] = [str(value) for value in normalized["pooling_types"]]
    normalized["skip_connections"] = [
        bool(value) for value in normalized["skip_connections"]
    ]
    normalized["dilations"] = [int(value) for value in normalized["dilations"]]
    normalized["separable"] = [bool(value) for value in normalized["separable"]]
    normalized["dropout"] = float(normalized["dropout"])
    return normalized


def mutate_genome(
    genome: Genome,
    search_space: dict[str, Any],
    rng: random.Random,
    mutation_rate: float,
) -> Genome:
    """Mutate a genome while keeping it inside the configured search space."""
    mutated = copy.deepcopy(genome)

    if rng.random() < mutation_rate:
        mutated["num_layers"] = rng.randint(
            int(search_space["num_layers"]["low"]),
            int(search_space["num_layers"]["high"]),
        )
        mutated = normalize_genome_layers(mutated, search_space, rng)

    for index in range(int(mutated["num_layers"])):
        if rng.random() < mutation_rate:
            mutated["filters"][index] = int(rng.choice(search_space["filters"]["choices"]))
        if rng.random() < mutation_rate:
            mutated["kernel_sizes"][index] = int(
                rng.choice(search_space["kernel_sizes"]["choices"])
            )
        if rng.random() < mutation_rate:
            mutated["pooling_types"][index] = str(
                rng.choice(search_space["pooling_types"]["choices"])
            )
        if rng.random() < mutation_rate:
            mutated["skip_connections"][index] = bool(
                rng.choice(search_space["skip_connections"]["choices"])
            )

    if rng.random() < mutation_rate:
        mutated["dropout"] = rng.uniform(
            float(search_space["dropout"]["low"]),
            float(search_space["dropout"]["high"]),
        )

    return normalize_genome_layers(mutated, search_space, rng)


def crossover_genomes(
    parent_a: Genome,
    parent_b: Genome,
    search_space: dict[str, Any],
    rng: random.Random,
) -> Genome:
    """Create a child genome by crossing over two parent genomes at a random layer index."""
    num_layers_a = int(parent_a["num_layers"])
    num_layers_b = int(parent_b["num_layers"])

    child = copy.deepcopy(parent_a if num_layers_a >= num_layers_b else parent_b)
    shorter = parent_b if num_layers_a >= num_layers_b else parent_a
    shorter_layers = int(shorter["num_layers"])

    if shorter_layers < 2:
        return normalize_genome_layers(copy.deepcopy(parent_a), search_space, rng)

    crossover_point = rng.randint(1, shorter_layers - 1)

    layer_fields = ["filters", "kernel_sizes", "pooling_types", "skip_connections", "dilations", "separable"]
    for field in layer_fields:
        child[field][:crossover_point] = copy.deepcopy(shorter[field][:crossover_point])

    child["filters"] = _generate_monotonic_filters(
        search_space, int(child["num_layers"]), rng
    )

    return normalize_genome_layers(child, search_space, rng)
