import torch
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
