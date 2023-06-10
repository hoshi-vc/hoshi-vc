# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# 論文 : https://arxiv.org/abs/1905.09263
# 参考にした実装 : https://github.com/ming024/FastSpeech2/blob/master/config/LJSpeech/model.yaml

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class PosFFT(nn.Module):
  """ PosEncoding + N * FFT Blocks """
  def __init__(self, iodim: int, layers: int, heads: int, hdim: int, kernels: tuple[int, int], dropout: float, posenc_len: int) -> None:
    super().__init__()
    self.pos_enc = PosEncoding(posenc_len, iodim)
    self.layers = nn.Sequential(*[FFTBlock(iodim, heads, hdim, kernels, dropout) for _ in range(layers)])

  def forward(self, x: Tensor) -> Tensor:
    out = x + self.pos_enc(x)
    out = self.layers(out)
    return out

class PosEncoding(nn.Module):
  def __init__(self, seq_len: int, odim: int) -> None:
    super().__init__()
    assert odim % 2 == 0
    self.odim = odim
    self.table: Tensor
    self.register_buffer("table", calc_posenc_table(seq_len, odim))

  def forward(self, x: Tensor):
    batch_size, length, _ = x.shape
    if length <= len(self.table):
      table = self.table[:length]
    else:
      print("PosEncoding: cache was not used")
      table = calc_posenc_table(length, self.odim, device=x.device)
    return table.unsqueeze(0).expand(batch_size, -1, -1)

def calc_posenc_table(length: int, dim: int, device=None):
  pos = torch.arange(length, device=device).unsqueeze(1)  # shape: [len, 1]
  term = pow_st(10000.0, torch.arange(0, dim, 2, device=device) / dim)  # shape: [dim//2]
  table = torch.zeros(length, dim, device=device)
  table[:, 0::2] = torch.sin(pos / term)
  table[:, 1::2] = torch.cos(pos / term)
  return table

def pow_st(base: float, exp: Tensor):
  """pow(scalar, tensor)"""
  return torch.exp(exp * math.log(base))

class FFTBlock(torch.nn.Module):
  def __init__(self, iodim: int, heads: int, hdim: int, kernels: tuple[int, int], dropout: float):
    super().__init__()
    self.attn = nn.MultiheadAttention(iodim, heads, dropout=dropout)
    self.norm1 = nn.LayerNorm(iodim)
    self.conv = PositionwiseFeedForward(iodim, hdim, kernels, dropout=dropout)
    self.norm2 = nn.LayerNorm(iodim)

  def forward(self, x: Tensor):
    residual = x
    out, _ = self.attn(x, x, x)
    out = out + residual
    out = self.norm1(out)

    residual = out
    out = self.conv(out)
    out = out + residual
    out = self.norm2(out)

    return out

class FFNBlock(torch.nn.Module):
  def __init__(self, iodim: int, hdim: int, kernels: tuple[int, int], dropout: float):
    super().__init__()
    self.conv = PositionwiseFeedForward(iodim, hdim, kernels, dropout=dropout)
    self.norm = nn.LayerNorm(iodim)

  def forward(self, x: Tensor):
    residual = x
    x = self.conv(x)
    x = x + residual
    x = self.norm(x)

    return x

class PositionwiseFeedForward(nn.Module):
  """ 2-layer 1D convolutional network with ReLU activation (Section 3.1) """
  def __init__(self, iodim: int, hdim: int, kernels: tuple[int, int], dropout: float):
    super().__init__()

    self.conv1 = nn.Conv1d(iodim, hdim, kernel_size=kernels[0], padding="same")
    self.conv2 = nn.Conv1d(hdim, iodim, kernel_size=kernels[1], padding="same")
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor):
    out = x.transpose(1, 2)
    out = self.conv1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = out.transpose(1, 2)
    out = self.dropout(out)

    return out
