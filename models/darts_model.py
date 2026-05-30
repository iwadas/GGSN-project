"""DARTS-inspired differentiable architecture search model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


OPS_NAMES = ["conv3x3", "conv5x5", "skip_connect", "max_pool_3x3", "avg_pool_3x3"]

OPS_TO_GENOME: dict[str, dict[str, Any]] = {
    "conv3x3": {"kernel_size": 3, "pooling_type": "max", "skip": False},
    "conv5x5": {"kernel_size": 5, "pooling_type": "max", "skip": False},
    "skip_connect": {"kernel_size": 3, "pooling_type": "max", "skip": True},
    "max_pool_3x3": {"kernel_size": 3, "pooling_type": "max", "skip": False},
    "avg_pool_3x3": {"kernel_size": 3, "pooling_type": "avg", "skip": False},
}


class ReLUConvBN(nn.Module):
    """Conv2d -> BatchNorm -> ReLU."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class MixedOp(nn.Module):
    """Mixed operation with learnable architecture weight alpha.

    Applies all candidate operations and returns a softmax-weighted sum.
    Each operation produces (B, C_out, H/2, W/2).
    """

    def __init__(self, C_in: int, C_out: int) -> None:
        super().__init__()
        self._ops = nn.ModuleList()

        for op_name in OPS_NAMES:
            if op_name == "conv3x3":
                op = nn.Sequential(
                    ReLUConvBN(C_in, C_out, 3, stride=1, padding=1),
                    nn.MaxPool2d(2),
                )
            elif op_name == "conv5x5":
                op = nn.Sequential(
                    ReLUConvBN(C_in, C_out, 5, stride=1, padding=2),
                    nn.MaxPool2d(2),
                )
            elif op_name == "skip_connect":
                if C_in == C_out:
                    op = nn.Sequential(nn.Identity(), nn.MaxPool2d(2))
                else:
                    op = nn.Sequential(
                        nn.Conv2d(C_in, C_out, 1, bias=False),
                        nn.BatchNorm2d(C_out),
                        nn.MaxPool2d(2),
                    )
            elif op_name == "max_pool_3x3":
                op = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True),
                )
            elif op_name == "avg_pool_3x3":
                op = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True),
                )
            else:
                raise ValueError(f"Unknown operation: {op_name}")
            self._ops.append(op)

        self.alpha = nn.Parameter(torch.zeros(len(self._ops)))

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        weights = F.softmax(self.alpha / temperature, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class DartsLayer(nn.Module):
    """One layer: MixedOp -> Dropout2d."""

    def __init__(self, C_in: int, C_out: int, dropout: float) -> None:
        super().__init__()
        self.mixed_op = MixedOp(C_in, C_out)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return self.dropout(self.mixed_op(x, temperature))


class DartsCNN(nn.Module):
    """CNN with DARTS-style mixed operations at each layer.

    Architecture parameters (alphas) are stored inside each MixedOp.
    """

    def __init__(
        self,
        filters: Sequence[int],
        dropout: float = 0.0,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if not filters:
            raise ValueError("filters must contain at least one layer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.filters = [int(f) for f in filters]
        self.temperature: float = 1.0

        layers: list[DartsLayer] = []
        in_channels = input_channels
        for out_channels in self.filters:
            layers.append(DartsLayer(in_channels, out_channels, dropout))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.filters[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, temperature=self.temperature)
        return self.classifier(x)

    def network_parameters(self):
        """Return all parameters EXCEPT architecture weights (alphas)."""
        arch_set = set(self.arch_parameters())
        return [p for p in self.parameters() if p not in arch_set]

    def arch_parameters(self):
        """Return the list of architecture weight tensors (one per layer)."""
        return [layer.mixed_op.alpha for layer in self.layers]


def derive_architecture(model: DartsCNN, dropout: float) -> dict[str, Any]:
    """Derive a discrete SearchCNN-compatible genome from trained alphas."""
    selected_op_names: list[str] = []
    for layer in model.layers:
        weights = F.softmax(layer.mixed_op.alpha, dim=0)
        idx = int(weights.argmax().item())
        selected_op_names.append(OPS_NAMES[idx])

    kernel_sizes: list[int] = []
    pooling_types: list[str] = []
    skip_connections: list[bool] = []
    for op_name in selected_op_names:
        mapping = OPS_TO_GENOME[op_name]
        kernel_sizes.append(mapping["kernel_size"])
        pooling_types.append(mapping["pooling_type"])
        skip_connections.append(mapping["skip"])

    return {
        "num_layers": len(selected_op_names),
        "filters": model.filters[:],
        "kernel_sizes": kernel_sizes,
        "pooling_types": pooling_types,
        "skip_connections": skip_connections,
        "dropout": dropout,
    }
