import torch
from torch import nn
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
