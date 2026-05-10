"""Configurable baseline CNN for CIFAR-10 experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """A compact configurable CNN baseline for 32x32 RGB images."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        filters: Sequence[int] = (32, 64, 128),
        kernel_sizes: int | Sequence[int] = 3,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if not filters:
            raise ValueError("filters must contain at least one layer width.")
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(filters)
        if len(kernel_sizes) != len(filters):
            raise ValueError("kernel_sizes must have the same length as filters.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(filters, kernel_sizes):
            padding = kernel_size // 2
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_baseline_cnn(
    num_layers: int = 3,
    base_filters: int = 32,
    filter_multiplier: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.2,
    num_classes: int = 10,
) -> BaselineCNN:
    """Build a baseline CNN from simple scalar hyperparameters."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    filters = [base_filters * (filter_multiplier**idx) for idx in range(num_layers)]
    kernel_sizes = [kernel_size] * num_layers
    return BaselineCNN(
        num_classes=num_classes,
        filters=filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    )
