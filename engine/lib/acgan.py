# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm

# ACGAN:     https://arxiv.org/abs/1610.09585
# PD-GAN:    https://arxiv.org/abs/1802.05637
# ContraGAN: https://arxiv.org/abs/2006.12681
# ADC-GAN:   https://arxiv.org/abs/2107.10060
# ReACGAN:   https://arxiv.org/abs/2111.01118
# ReACGAN 日本語解説: https://ai-scholar.tech/articles/gan/reacgan
# 様々な GAN の実装: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN

class BasicDiscriminator(nn.Module):
  # Based on GANSpeech :: https://arxiv.org/pdf/2106.15153.pdf
  # Conv1d に spectral_norm をかけた。
  # Conditioning は取り除いた。
  def __init__(self, dims: list[int], kernels: list[int], strides: list[int], use_spectral_norm: bool, avg_pool: Optional[nn.Module] = None):
    super().__init__()
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm

    # dims: (hdim1, hdim2, ..., odim)
    assert len(dims) == len(kernels) == len(strides)

    if avg_pool is None: avg_pool = nn.AdaptiveAvgPool2d(1)  # 1x1 の画像にまとめる

    dims = [1] + dims  # (idim, hdim1, hdim2, ..., odim)

    blocks: list[nn.Module] = []
    for i in range(len(dims) - 1):
      last = i == len(dims) - 2

      if not last:
        modules = [
            norm_f(nn.Conv2d(dims[i], dims[i + 1], kernels[i], strides[i], kernels[i] // 2)),
            # nn.InstanceNorm2d(dims[i + 1]),
            # nn.BatchNorm2d(dims[i + 1]),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
        ]
      else:
        modules = [
            avg_pool,
            norm_f(nn.Conv2d(dims[i], dims[i + 1], kernels[i], strides[i], kernels[i] // 2)),
        ]

      blocks += [nn.Sequential(*modules)]

    self.blocks = nn.ModuleList(blocks)
    self.odim = dims[-1]

  def forward(self, x: Tensor):
    # x: (batch, time, freq)

    x = x.unsqueeze(1)  # (batch, 1, time, freq)

    features = []
    for module in self.blocks:
      x = module(x)
      features.append(x)

    features, out = features[:-1], features[-1]  # out: (batch, odim, h, w)
    return out, features

class ACDiscriminator(nn.Module):
  def __init__(self, base: BasicDiscriminator, n_class: int, norm_feats: bool):
    super().__init__()
    self.base = base
    self.aux_linear = nn.Conv2d(base.odim, n_class, 1)
    self.norm_feats = norm_feats

  def forward(self, x: Tensor):
    # x: (batch, time, freq)
    # c: (batch, n_class, h, w)

    # out: (batch, odim)
    out, features = self.base(x)

    features += [out]

    # Feature Normalization :: Proposed by ReACGAN
    # TODO: ReACGAN では self.aux_linear の weights も normalize している
    #       ただ、公式実装を見ると、バグがあって weights の normalize はできてない
    #       （ self.linear2.parameters() の normalize したあとのテンソルをパラメーターにセットできてない :: Python にポインターはないので ）
    #       だから別にこのままでいいかなって思ってる。（勾配を維持しながら値だけ書き換えるとか、できるのかもしていいのかもわからないし）
    #       ref: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/e95bcd46372573581ae8b34c083e65bd5e4e0e9e/src/models/big_resnet.py#L383
    if self.norm_feats:
      out = F.normalize(out, dim=1)  # 超球面上に投影

    c = self.aux_linear(out)

    return c, features

def aux_loss(c: Tensor, s: Tensor):
  # c: (batch, n_class, h, w)
  # s: (batch, 1) :: 0 ~ n_class - 1

  c = torch.log_softmax(c, dim=1)
  s = s.unsqueeze(-1).expand(-1, c.size(2), c.size(3))  # -> (batch, h, w)
  loss = F.nll_loss(c, s)

  return loss
