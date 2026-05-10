"""CNN builder used by architecture search algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SearchConvBlock(nn.Module):
    """Convolutional block with optional residual projection and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pooling_type: str,
        use_skip: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.use_skip = use_skip
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if use_skip and in_channels != out_channels
            else None
        )

        if pooling_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2)
        elif pooling_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.batch_norm(self.conv(x)))
        if self.use_skip:
            if self.projection is not None:
                residual = self.projection(residual)
            out = out + residual
        out = self.pool(out)
        return self.dropout(out)


class SearchCNN(nn.Module):
    """CNN created from a NAS genome."""

    def __init__(
        self,
        filters: Sequence[int],
        kernel_sizes: Sequence[int],
        pooling_types: Sequence[str],
        skip_connections: Sequence[bool],
        dropout: float,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        layer_count = len(filters)
        if layer_count == 0:
            raise ValueError("filters must contain at least one layer.")
        if not (
            len(kernel_sizes)
            == len(pooling_types)
            == len(skip_connections)
            == layer_count
        ):
            raise ValueError("All per-layer genome fields must have the same length.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size, pooling_type, use_skip in zip(
            filters,
            kernel_sizes,
            pooling_types,
            skip_connections,
        ):
            blocks.append(
                SearchConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pooling_type=pooling_type,
                    use_skip=use_skip,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_search_cnn_from_genome(
    genome: dict,
    num_classes: int = 10,
) -> SearchCNN:
    """Build a search CNN from a serializable genome dictionary."""
    return SearchCNN(
        filters=list(genome["filters"]),
        kernel_sizes=list(genome["kernel_sizes"]),
        pooling_types=list(genome["pooling_types"]),
        skip_connections=list(genome["skip_connections"]),
        dropout=float(genome["dropout"]),
        num_classes=num_classes,
    )
