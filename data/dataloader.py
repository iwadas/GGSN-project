"""CIFAR-10 dataset and dataloader factory functions."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

from data.transforms import get_eval_transforms, get_train_transforms


def get_cifar10_datasets(
    data_dir: str | Path = "data/raw",
    validation_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Download CIFAR-10 if needed and return train/validation/test datasets."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    data_path = Path(data_dir)
    train_eval_dataset = CIFAR10(
        root=str(data_path),
        train=True,
        transform=get_eval_transforms(),
        download=download,
    )
    test_dataset = CIFAR10(
        root=str(data_path),
        train=False,
        transform=get_eval_transforms(),
        download=download,
    )

    validation_size = int(len(train_eval_dataset) * validation_ratio)
    train_size = len(train_eval_dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_dataset = random_split(
        train_eval_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_augmented_dataset = CIFAR10(
        root=str(data_path),
        train=True,
        transform=get_train_transforms(),
        download=False,
    )
    train_dataset = torch.utils.data.Subset(
        train_augmented_dataset,
        train_subset.indices,
    )

    return train_dataset, validation_dataset, test_dataset


def get_cifar10_dataloaders(
    data_dir: str | Path = "data/raw",
    batch_size: int = 64,
    validation_ratio: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/validation/test dataloaders for CIFAR-10."""
    train_dataset, validation_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        validation_ratio=validation_ratio,
        seed=seed,
        download=download,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader
