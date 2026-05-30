"""Image transforms for CIFAR-10 experiments."""

from __future__ import annotations

import random

import torch
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Randomly mask a square region of the input tensor."""

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        y = random.randint(0, h)
        x = random.randint(0, w)
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        img[..., y1:y2, x1:x2] = 0.0
        return img


def get_train_transforms(
    random_crop: bool = True,
    random_horizontal_flip: bool = True,
    cutout_size: int = 16,
) -> transforms.Compose:
    """Return CIFAR-10 training transforms with optional augmentation."""
    transform_steps = []

    if random_crop:
        transform_steps.append(transforms.RandomCrop(32, padding=4))
    if random_horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip())

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    if cutout_size > 0:
        transform_steps.append(Cutout(size=cutout_size))

    return transforms.Compose(transform_steps)


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic CIFAR-10 validation/test transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
