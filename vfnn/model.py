"""The vfnn network: a small pre-activation residual 1-D CNN with a pooled binary head.

Follows ``vfnn/DESIGN.md`` section 3. The structure comes from the descriptions in Hannun
et al. 2019 (and its preprint, Rajpurkar et al. 2017) and Nasimi and Yazdchi 2022, not from
the GPL reference code. Input is ``(batch, 1, samples)``; output is one logit per window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    """Architecture knobs. Defaults are B-lite from the design."""
    channels: tuple[int, ...] = (16, 16, 32, 32, 64, 64)   # filters per residual block
    strides: tuple[int, ...] = (2, 2, 2, 2, 1, 1)          # stride of each block's first conv
    kernel: int = 15                                       # odd, so padding is symmetric
    stem: int = 16                                         # filters of the first conv
    dropout: float = 0.2
pass #class


B_LITE = Config()
B_MID = Config(channels=(32, 32, 32, 64, 64, 64), stem=32)


class Block(nn.Module):
    """Pre-activation residual block: two convolutions, max-pool shortcut.

    The first BN-ReLU is skipped in the first block because the stem already applied one.
    When the filter count grows, the shortcut is zero-padded along channels, so it adds no
    parameters.
    """

    def __init__(self, cin: int, cout: int, stride: int, cfg: Config, first: bool = False):
        super().__init__()
        pad = cfg.kernel // 2
        self.pre1 = nn.Identity() if first else nn.Sequential(nn.BatchNorm1d(cin), nn.ReLU())
        self.conv1 = nn.Conv1d(cin, cout, cfg.kernel, stride=stride, padding=pad)
        self.pre2 = nn.Sequential(nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.conv2 = nn.Conv1d(cout, cout, cfg.kernel, padding=pad)
        self.pool = nn.MaxPool1d(stride, ceil_mode=True) if stride > 1 else nn.Identity()
        self.extra = cout - cin
    pass #def

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        if self.extra > 0:
            s = nn.functional.pad(s, (0, 0, 0, self.extra))
        pass #if
        y = self.conv1(self.pre1(x))
        y = self.conv2(self.pre2(y))
        return s + y
    pass #def
pass #class


class VFNet(nn.Module):
    """Stem, residual blocks, BN-ReLU, global average and max pooling, one logit."""

    def __init__(self, cfg: Config = B_LITE):
        super().__init__()
        if len(cfg.channels) != len(cfg.strides):
            raise ValueError("channels and strides must have the same length")
        pass #if
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv1d(1, cfg.stem, cfg.kernel, padding=cfg.kernel // 2),
            nn.BatchNorm1d(cfg.stem),
            nn.ReLU(),
        )
        blocks = []
        cin = cfg.stem
        for i, (c, s) in enumerate(zip(cfg.channels, cfg.strides)):
            blocks.append(Block(cin, c, s, cfg, first=(i == 0)))
            cin = c
        pass #for
        self.blocks = nn.Sequential(*blocks)
        self.post = nn.Sequential(nn.BatchNorm1d(cin), nn.ReLU())
        self.head = nn.Linear(2 * cin, 1)
        self.apply(self._init)
    pass #def

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        pass #if
    pass #def

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.post(self.blocks(self.stem(x)))
        z = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(z).squeeze(1)
    pass #def
pass #class


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
pass #def


def macs(model: nn.Module, samples: int) -> int:
    """Multiply-accumulates of the convolutions and the dense head for one window."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv1d):
            total += out.numel() * m.in_channels * m.kernel_size[0] // out.shape[0]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
        pass #if
    pass #def

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (nn.Conv1d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, samples))
    pass #with
    model.train(was)
    for h in hooks:
        h.remove()
    pass #for
    return total
pass #def


def receptive_field(cfg: Config) -> int:
    """Receptive field in input samples of one output position of the last block."""
    k = cfg.kernel
    rf, jump = k, 1
    for s in cfg.strides:
        rf += (k - 1) * jump
        jump *= s
        rf += (k - 1) * jump
    pass #for
    return rf
pass #def
