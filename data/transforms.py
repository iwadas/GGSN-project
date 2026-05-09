"""Image transforms for CIFAR-10 experiments."""

from __future__ import annotations

from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transforms(
    random_crop: bool = True,
    random_horizontal_flip: bool = True,
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
    return transforms.Compose(transform_steps)


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic CIFAR-10 validation/test transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
