# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

class GradientReversalFn(Function):
  @staticmethod
  def forward(ctx, x, scale):
    ctx.save_for_backward(scale)
    return x

  @staticmethod
  def backward(ctx, grad_out):
    scale, = ctx.saved_tensors
    return -scale * grad_out, None

class GradientReversal(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = torch.tensor(scale, requires_grad=False)

  def forward(self, x):
    return GradientReversalFn.apply(x, self.scale)

  def update_scale(self, scale: float):
    if self.scale != scale:
      self.scale = torch.tensor(scale, requires_grad=False)

class Transpose(nn.Module):
  def __init__(self, dim0: int, dim1: int):
    super().__init__()
    self.dim0 = dim0
    self.dim1 = dim1

  def forward(self, x: Tensor) -> Tensor:
    return x.transpose(self.dim0, self.dim1)

class GetNth(nn.Module):
  def __init__(self, n: int):
    super().__init__()
    self.n = n

  def forward(self, x: Tensor) -> Tensor:
    return x[self.n]

class Buckets(nn.Module):
  def __init__(self, min_value: float, max_value: float, n_bins: int):
    super().__init__()
    bins = torch.linspace(min_value, max_value, n_bins - 1)  # 並木算的な理由で bins - 1

    self.bins = nn.parameter.Parameter(bins, requires_grad=False)

  def forward(self, o: Tensor):
    return torch.bucketize(o, self.bins)

# (batch, channels, height, time)
class DownSample(nn.Module):
  def __init__(self, layer_type: str):
    super().__init__()
    self.layer_type = layer_type

  def forward(self, x: Tensor):
    if self.layer_type == 'none':
      return x
    elif self.layer_type == 'time-preserve':
      assert x.shape[2] % 2 == 0, f'x.shape[2] is {x.shape[2]}'
      return F.avg_pool2d(x, (2, 1))
    elif self.layer_type == 'half':
      assert x.shape[2] % 2 == 0, f'x.shape[2] is {x.shape[2]}'
      assert x.shape[3] % 2 == 0, f'x.shape[3] is {x.shape[3]}'
      return F.avg_pool2d(x, 2)
    else:
      raise RuntimeError(f'Unexpected: {self.layer_type}')

# (batch, channels, height, time)
class UpSample(nn.Module):
  def __init__(self, layer_type: str, amount=2):
    super().__init__()
    self.layer_type = layer_type
    self.amount = amount

  def forward(self, x: Tensor):
    if self.layer_type == 'none':
      return x
    elif self.layer_type == 'time-preserve':
      return F.interpolate(x, scale_factor=(self.amount, 1), mode='nearest')
    elif self.layer_type == 'half':
      return F.interpolate(x, scale_factor=self.amount, mode='nearest')
    else:
      raise RuntimeError(f'Unexpected: {self.layer_type}')

# (batch, channels, height, time)
class ResDown(nn.Module):
  """ Residual block for downsampling. """
  def __init__(self, idim: int, odim: int, normalize: bool, downsample='none'):
    super().__init__()

    self.conv1x1 = nn.Conv2d(idim, odim, 1, 1, 0, bias=False) if idim != odim else None
    self.downsample = DownSample(downsample)

    self.conv1 = nn.Conv2d(idim, idim, 3, 1, 1)
    self.conv2 = nn.Conv2d(idim, odim, 3, 1, 1)
    self.norm1 = nn.InstanceNorm2d(idim, affine=True) if normalize else None
    self.norm2 = nn.InstanceNorm2d(idim, affine=True) if normalize else None

  def _shortcut(self, x: Tensor):
    if self.conv1x1: x = self.conv1x1(x)
    x = self.downsample(x)
    return x

  def _residual(self, x: Tensor):
    if self.norm1: x = self.norm1(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv1(x)
    x = self.downsample(x)
    if self.norm2: x = self.norm2(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv2(x)
    return x

  def forward(self, x: Tensor):
    x = self._shortcut(x) + self._residual(x)
    return x / math.sqrt(2)  # unit variance

# (batch, channels, height, time)
class ResUp(nn.Module):
  """ Residual block for upsampling. """
  def __init__(self, idim: int, odim: int, normalize: bool, upsample='none'):
    super().__init__()

    self.conv1x1 = nn.Conv2d(idim, odim, 1, 1, 0, bias=False) if idim != odim else None
    self.upsample = UpSample(upsample)

    self.conv1 = nn.Conv2d(idim, idim, 3, 1, 1)
    self.conv2 = nn.Conv2d(idim, odim, 3, 1, 1)
    self.norm1 = nn.InstanceNorm2d(idim, affine=True) if normalize else None
    self.norm2 = nn.InstanceNorm2d(idim, affine=True) if normalize else None

  def _shortcut(self, x: Tensor):
    x = self.upsample(x)
    if self.conv1x1: x = self.conv1x1(x)
    return x

  def _residual(self, x: Tensor):
    if self.norm1: x = self.norm1(x)
    x = F.leaky_relu(x, 0.2)
    x = self.upsample(x)
    x = self.conv1(x)
    if self.norm2: x = self.norm2(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv2(x)
    return x

  def forward(self, x: Tensor):
    x = self._shortcut(x) + self._residual(x)
    return x / math.sqrt(2)  # unit variance
