# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import torch.nn.functional as F
from torch import Tensor, nn

class CondNorm1(nn.Module):
  def __init__(self, hdim: int, cdim: int):
    super().__init__()
    self.bias = nn.Linear(cdim, hdim)

  def forward(self, seq: Tensor, c: Tensor) -> Tensor:
    bias = self.bias(c).unsqueeze(1)
    seq = seq + bias

    return seq

class CondNorm2(nn.Module):
  def __init__(self, hdim: int, cdim: int):
    super().__init__()
    self.scale = nn.Linear(cdim, hdim)
    self.bias = nn.Linear(cdim, hdim)

  def forward(self, seq: Tensor, c: Tensor) -> Tensor:
    bias = self.bias(c).unsqueeze(1)
    scale = self.scale(c).unsqueeze(1)
    seq = seq * scale.exp() + bias

    return seq

class CondNorm3(nn.Module):
  def __init__(self, hdim: int, cdim: int):
    super().__init__()
    self.scale = nn.Linear(cdim, hdim)
    self.bias = nn.Linear(cdim, hdim)
    self.softsign = nn.Linear(cdim, hdim)

  def forward(self, seq: Tensor, c: Tensor) -> Tensor:
    scale = self.scale(c).unsqueeze(1)
    bias = self.bias(c).unsqueeze(1)
    softsign = self.softsign(c).unsqueeze(1)
    seq = seq * scale.exp() + bias + F.softsign(seq) * softsign

    return seq
