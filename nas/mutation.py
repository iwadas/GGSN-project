"""Genome creation and mutation for evolutionary NAS."""

from __future__ import annotations

import copy
import random
from typing import Any


Genome = dict[str, Any]


def random_genome(search_space: dict[str, Any], rng: random.Random) -> Genome:
    """Create one random architecture genome from a search-space config."""
    num_layers = rng.randint(
        int(search_space["num_layers"]["low"]),
        int(search_space["num_layers"]["high"]),
    )
    return {
        "num_layers": num_layers,
        "filters": [
            int(rng.choice(search_space["filters"]["choices"]))
            for _ in range(num_layers)
        ],
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

    normalized["filters"] = [int(value) for value in normalized["filters"]]
    normalized["kernel_sizes"] = [int(value) for value in normalized["kernel_sizes"]]
    normalized["pooling_types"] = [str(value) for value in normalized["pooling_types"]]
    normalized["skip_connections"] = [
        bool(value) for value in normalized["skip_connections"]
    ]
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
