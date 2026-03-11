from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class ECTFormerConfig:
    widths: Sequence[int] = (40, 80, 160, 320)
    depths: Sequence[int] = (2, 2, 8, 2)
    stem_patch: int = 4
    patch_expand_ratio: int = 4
    conv_expand_ratio: int = 4
    attn_expand_ratio: int = 4
    num_classes: int = 1000
    layer_scale_init_value: float = 1e-6
    drop_rate: float = 0.0


def compute_dynamic_kernels(strides: Sequence[int], dk0: int = 3) -> List[int]:
    """
    Implements Eq. (1)~(3) in the paper.
    Example: strides=(4,2,2,2), dk0=3 -> [3,5,5,7]
    """
    kernels = [dk0]
    current = dk0
    accum = 1
    for i, s in enumerate(strides, start=1):
        accum *= s
        asi = int(round(math.log2(accum)))
        alpha = int((asi % 2) and (asi // 2))
        if i == 1:
            kernels[0] = current
        else:
            current = current + alpha * 2
            kernels.append(current)
    if len(kernels) == 1:
        kernels = [dk0]
    return kernels


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, 1, 1)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
        bias: bool = False,
    ):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=bias
            ),
            nn.BatchNorm2d(out_chs),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class Stem(nn.Module):
    """
    Small-patch stem. We keep a 4x4 / stride-4 patchifying stem, consistent with the
    paper's stage-1 downsampling ratio (4,2,2,2) and ConvNeXt-style design.
    """

    def __init__(self, in_chans: int, out_chs: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                out_chs,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_chs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvModule(nn.Module):
    """
    ConvNeXt-style block with BN + SiLU and dynamic-kernel depthwise convolution.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        expand_ratio: int = 4,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.layer_scale = LayerScale(dim, layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        return x + shortcut


class IBOPF(nn.Module):
    """
    Inverted Bottleneck Overlapping Patchify, Eq. (4).
    x_hat = Conv1x1(DWConv3x3_{stride=patch_size}(Conv1x1(x)))

    We use expansion ratio epsilon=4 as suggested by the paper.
    """

    def __init__(
        self, in_chs: int, out_chs: int, patch_size: int = 2, expand_ratio: int = 4
    ):
        super().__init__()
        hidden = in_chs * expand_ratio
        self.block = nn.Sequential(
            ConvBNAct(in_chs, hidden, kernel_size=1, stride=1, padding=0, act=True),
            ConvBNAct(
                hidden,
                hidden,
                kernel_size=3,
                stride=patch_size,
                padding=1,
                groups=hidden,
                act=True,
            ),
            ConvBNAct(hidden, out_chs, kernel_size=1, stride=1, padding=0, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSASingleHeadAttention(nn.Module):
    """
    Dimension-Split Single-Head Self-Attention (DSA).

    Faithful to the paper's key ideas:
    - Q and K are produced by DWConv + BN + SiLU, then split along channels.
    - V uses PEG-like depthwise conv with residual: DWConv(x) + x.
    - Single-head attention to avoid multi-head reshape overhead.

    Notes:
    - The paper does not publish every low-level implementation detail (e.g. exact
      projection ordering in released code was not available to us), so this is a
      best-faithful implementation based on Fig. 2/3 and Eqs. (5)~(8).
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        proj_expand_ratio: int = 4,
        layer_scale_init_value: float = 1e-6,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.qk = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.peg = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        hidden = dim * proj_expand_ratio
        # Projection branch approximated from Fig. 2: 1x1 -> BN -> 1x1 -> SiLU -> 1x1
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.layer_scale = LayerScale(dim, layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        b, c, h, w = x.shape
        qk = self.qk(x)
        q, k = torch.split(qk, [c // 2, c - c // 2], dim=1)
        v = self.peg(x) + x

        q = q.flatten(2).transpose(1, 2)  # [B, HW, Cq]
        k = k.flatten(2).transpose(1, 2)  # [B, HW, Ck]
        v = v.flatten(2).transpose(1, 2)  # [B, HW, C]

        scale = math.sqrt(float(c))
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        z = torch.matmul(attn, v)  # [B, HW, C]
        z = z.transpose(1, 2).reshape(b, c, h, w)
        z = self.proj(z)
        z = self.layer_scale(z)
        return z + shortcut


class ECTStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int,
        use_dsa_last: bool,
        conv_expand_ratio: int,
        attn_expand_ratio: int,
        layer_scale_init_value: float,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            is_last = i == depth - 1
            if is_last and use_dsa_last:
                blocks.append(
                    DSASingleHeadAttention(
                        dim=dim,
                        kernel_size=kernel_size,
                        proj_expand_ratio=attn_expand_ratio,
                        layer_scale_init_value=layer_scale_init_value,
                        attn_drop=drop_rate,
                    )
                )
            else:
                blocks.append(
                    ConvModule(
                        dim=dim,
                        kernel_size=kernel_size,
                        expand_ratio=conv_expand_ratio,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ECTFormer(nn.Module):
    def __init__(self, config: ECTFormerConfig):
        super().__init__()
        self.config = config
        widths = list(config.widths)
        depths = list(config.depths)
        assert len(widths) == 4 and len(depths) == 4, "ECTFormer expects 4 stages."
        kernels = compute_dynamic_kernels((4, 2, 2, 2), dk0=3)
        # kernels is length 4: [3,5,5,7]
        if len(kernels) != 4:
            kernels = [3, 5, 5, 7]

        self.stem = Stem(3, widths[0], patch_size=config.stem_patch)

        self.stage1 = ECTStage(
            dim=widths[0],
            depth=depths[0],
            kernel_size=kernels[0],
            use_dsa_last=False,
            conv_expand_ratio=config.conv_expand_ratio,
            attn_expand_ratio=config.attn_expand_ratio,
            layer_scale_init_value=config.layer_scale_init_value,
            drop_rate=config.drop_rate,
        )
        self.down1 = IBOPF(
            widths[0], widths[1], patch_size=2, expand_ratio=config.patch_expand_ratio
        )
        self.stage2 = ECTStage(
            dim=widths[1],
            depth=depths[1],
            kernel_size=kernels[1],
            use_dsa_last=True,
            conv_expand_ratio=config.conv_expand_ratio,
            attn_expand_ratio=config.attn_expand_ratio,
            layer_scale_init_value=config.layer_scale_init_value,
            drop_rate=config.drop_rate,
        )
        self.down2 = IBOPF(
            widths[1], widths[2], patch_size=2, expand_ratio=config.patch_expand_ratio
        )
        self.stage3 = ECTStage(
            dim=widths[2],
            depth=depths[2],
            kernel_size=kernels[2],
            use_dsa_last=True,
            conv_expand_ratio=config.conv_expand_ratio,
            attn_expand_ratio=config.attn_expand_ratio,
            layer_scale_init_value=config.layer_scale_init_value,
            drop_rate=config.drop_rate,
        )
        self.down3 = IBOPF(
            widths[2], widths[3], patch_size=2, expand_ratio=config.patch_expand_ratio
        )
        self.stage4 = ECTStage(
            dim=widths[3],
            depth=depths[3],
            kernel_size=kernels[3],
            use_dsa_last=True,
            conv_expand_ratio=config.conv_expand_ratio,
            attn_expand_ratio=config.attn_expand_ratio,
            layer_scale_init_value=config.layer_scale_init_value,
            drop_rate=config.drop_rate,
        )

        self.norm = nn.BatchNorm2d(widths[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], config.num_classes)
        self.dropout = (
            nn.Dropout(config.drop_rate) if config.drop_rate > 0 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.norm(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.head(x)


def build_ectformer(
    variant: str = "x1.0",
    num_classes: int = 1000,
    drop_rate: float = 0.0,
) -> ECTFormer:
    base_widths = [40, 80, 160, 320]
    depths = [2, 2, 8, 2]

    mapping = {
        "x1.0": 1.0,
        "x1.25": 1.25,
        "x1.5": 1.5,
    }
    if variant not in mapping:
        raise ValueError(
            f"Unknown variant: {variant}. Expected one of {list(mapping.keys())}."
        )

    width_mult = mapping[variant]
    widths = [int(round(w * width_mult)) for w in base_widths]

    cfg = ECTFormerConfig(
        widths=widths,
        depths=depths,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return ECTFormer(cfg)


if __name__ == "__main__":
    model = build_ectformer("x1.0", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {total:.2f}M")
