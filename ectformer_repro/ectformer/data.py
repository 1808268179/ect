from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_root: str
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


def build_train_transform(
    image_size: int = 224, randaugment: bool = True, random_erasing: bool = True
):
    ops = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
    ]
    if randaugment:
        ops.append(transforms.RandAugment())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if random_erasing:
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)


def build_eval_transform(image_size: int = 224):
    resize_size = int(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_datasets(
    data_root: str,
    image_size: int = 224,
    randaugment: bool = True,
    random_erasing: bool = True,
):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"val directory not found: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test directory not found: {test_dir}")

    train_ds = datasets.ImageFolder(
        train_dir,
        transform=build_train_transform(
            image_size=image_size,
            randaugment=randaugment,
            random_erasing=random_erasing,
        ),
    )
    val_ds = datasets.ImageFolder(
        val_dir, transform=build_eval_transform(image_size=image_size)
    )
    test_ds = datasets.ImageFolder(
        test_dir, transform=build_eval_transform(image_size=image_size)
    )

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError(
            "train/val/test class folders are inconsistent. "
            "Please ensure the same class names exist in all splits."
        )

    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg: DataConfig, randaugment: bool = True, random_erasing: bool = True
):
    train_ds, val_ds, test_ds = build_datasets(
        data_root=cfg.data_root,
        image_size=cfg.image_size,
        randaugment=randaugment,
        random_erasing=random_erasing,
    )

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, test_loader, train_ds.classes
