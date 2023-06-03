# %%
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

# CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information
# Paper: https://arxiv.org/abs/2006.12013
# Code:  https://github.com/Linear95/CLUB
class CLUBSample(nn.Module):
  def __init__(self, xdim: int, ydim: int, hdim: int):
    super().__init__()
    hdim = hdim // 2
    self.mu = nn.Sequential(
        nn.Linear(xdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, ydim),
    )
    self.logvar = nn.Sequential(
        nn.Linear(xdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, ydim),
        nn.Tanh(),
    )

  def log_likelihood(self, xs: Tensor, ys: Tensor):
    """
    Args:
      xs: (batch, xdim)
      ys: (batch, ydim)
    """
    mu = self.mu(xs)
    logvar = self.logvar(xs)

    return (-(mu - ys)**2 / logvar.exp() - logvar).sum(dim=-1).mean()

  def forward(self, xs: Tensor, ys: Tensor):
    """
    Estimates the mutual information between x and y.

    Args:
      xs: (batch, xdim)
      ys: (batch, ydim)
    """
    mu = self.mu(xs)
    logvar = self.logvar(xs)

    # それぞれの x に対してランダムに一つの y を選ぶ
    y_samples = ys[torch.randperm(xs.shape[0])]

    pos = -(mu - ys)**2 / logvar.exp()
    neg = -(mu - y_samples)**2 / logvar.exp()
    upper_bound = (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()
    return upper_bound / 2.

  def learning_loss(self, xs: Tensor, ys: Tensor):
    """
    Args:
      xs: (batch, xdim)
      ys: (batch, ydim)
    """
    return -self.log_likelihood(xs, ys)

def check_club():
  # 公式に提供されている学習例と比べることで、処理が変わっていないことを確認する
  # https://github.com/Linear95/CLUB/blob/31ea373c2d316221d40ae63a462bf964930f2117/mi_estimation.ipynb

  import matplotlib.pyplot as plt
  import numpy as np
  from tqdm import tqdm

  def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result

  def sample(rho=0.5, dim=20, batch_size=128):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T
    x = torch.as_tensor(x, dtype=torch.float32).reshape(batch_size, dim)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(batch_size, dim)
    return x, y

  sample_dim = 20
  steps = 4000
  mi_list = [2.0, 4.0, 6.0, 8.0, 10.0]

  model = CLUBSample(sample_dim, sample_dim, 15)
  optimizer = torch.optim.Adam(model.parameters(), 5e-3)

  mi_est = []
  for mi in mi_list:
    rho = mi_to_rho(mi, sample_dim)
    for _ in tqdm(range(steps)):
      batch_x, batch_y = sample(rho, dim=sample_dim, batch_size=64)

      model.eval()
      mi_est.append(model(batch_x, batch_y).item())

      model.train()
      loss = model.learning_loss(batch_x, batch_y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  plt.plot(mi_est, alpha=0.4)
  plt.plot(np.repeat(mi_list, steps))

if __name__ == "__main__":
  check_club()

# %%
