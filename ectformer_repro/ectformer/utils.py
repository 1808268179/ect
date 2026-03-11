from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class CosineScheduler:
    def __init__(
        self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int):
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if epoch < self.warmup_epochs:
                lr = base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
            else:
                progress = (epoch - self.warmup_epochs) / float(
                    max(1, self.total_epochs - self.warmup_epochs)
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            group["lr"] = lr


class SoftTargetCrossEntropy(torch.nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = x.size(-1)
        log_preds = F.log_softmax(x, dim=-1)
        loss = -log_preds.sum(dim=-1).mean() * self.smoothing / n
        nll = F.nll_loss(log_preds, target)
        return loss + (1.0 - self.smoothing) * nll


def one_hot(target: torch.Tensor, num_classes: int, on_value=1.0, off_value=0.0):
    x = torch.full(
        (target.size(0), num_classes),
        off_value,
        device=target.device,
        dtype=torch.float,
    )
    x.scatter_(1, target.view(-1, 1), on_value)
    return x


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 1.0,
):
    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or random.random() > prob:
        return images, one_hot(targets, num_classes), False

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    indices = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
        )
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = images * lam + shuffled_images * (1.0 - lam)

    target_a = one_hot(targets, num_classes)
    target_b = one_hot(shuffled_targets, num_classes)
    mixed_targets = target_a * lam + target_b * (1.0 - lam)
    return images, mixed_targets, True


def save_json(obj: Any, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
