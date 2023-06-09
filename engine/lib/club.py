# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# NOTE
# プロファイリングしたら randperm に相当な時間（全体の 30% ）がかかってた。
# より高速な randint で代用しようと思ったけど、一番下の toy example でもすでに結果が違ってたから大丈夫かわからん。
# まぁ、推定された MI の値は異なるけど MI と比例してる感じだったから大丈夫かな？
# 論文 (Section 3.3) でも "we randomly sample a negative pair (xi, yki) [...], with ki uniformly selected from indices {1, 2, ... , N}." って言ってるし。

# CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information
# Paper: https://arxiv.org/abs/2006.12013
# Code:  https://github.com/Linear95/CLUB
class CLUBSampleInSeq(nn.Module):
  """
  ネガティブサンプルを取ってくるときに、同じサンプル列内から取ってくるようにしたもの

  xs: (batch, samples, xdim)
  ys: (batch, samples, ydim)
  """
  def __init__(self, xdim: int, ydim: int, hdim: int, fast_sampling=False):
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
    self.fast_sampling = fast_sampling

  def _sample_negatives(self, xs: Tensor, ys: Tensor):
    # それぞれの x に対してランダムに一つの y を選ぶ
    # このとき、同じシーケンス内から取ってくるようにする
    # 実装しやすいから、バッチ内の異なるシーケンスでも x, y のインデックスの対応は共通にしたけど多分大丈夫でしょ！
    # 長い音声からランダムな場所を切り抜いてデータセット作ってるし。
    size = xs.shape[1]
    if self.fast_sampling:
      random_index = torch.randint(size, (size,), device=xs.device).long()
    else:
      random_index = torch.randperm(size)
    return ys[:, random_index, :]

  def log_likelihood(self, xs: Tensor, ys: Tensor):
    mu = self.mu(xs)
    logvar = self.logvar(xs)

    return (-(mu - ys)**2 / logvar.exp() - logvar).sum(dim=-1).mean()

  def forward(self, xs: Tensor, ys: Tensor):
    """Estimates the mutual information between x and y. """
    mu = self.mu(xs)
    logvar = self.logvar(xs)

    # それぞれの x に対してランダムに一つの y を選ぶ
    ns = self._sample_negatives(xs, ys)

    pos = -(mu - ys)**2 / logvar.exp()
    neg = -(mu - ns)**2 / logvar.exp()
    upper_bound = (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()
    return upper_bound / 2.

  def learning_loss(self, xs: Tensor, ys: Tensor):
    return -self.log_likelihood(xs, ys)

class CLUBSample(CLUBSampleInSeq):
  """
  xs: (...samples, xdim)
  ys: (...samples, ydim)
  """
  def _sample_negatives(self, xs: Tensor, ys: Tensor):
    size = xs.shape[0]
    if self.fast_sampling:
      random_index = torch.randint(size, (size,), device=xs.device).long()
    else:
      random_index = torch.randperm(size)
    return ys[random_index]

  def log_likelihood(self, xs: Tensor, ys: Tensor):
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1, ys.shape[-1])
    return super().log_likelihood(xs, ys)

  def forward(self, xs: Tensor, ys: Tensor):
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1, ys.shape[-1])
    return super().forward(xs, ys)

class CLUBSampleForCategorical(nn.Module):
  '''
  xs: (...samples, xdim)
  ys: (...samples,)
  '''
  def __init__(self, xdim: int, ynum: int, hdim: int, fast_sampling=False, logvar: Optional[nn.Module] = None):
    super().__init__()

    if logvar is None:
      logvar = nn.Sequential(
          nn.Linear(xdim, hdim),
          nn.ReLU(),
          nn.Linear(hdim, hdim),
          nn.ReLU(),
          nn.Linear(hdim, ynum),
      )

    self.ynum = ynum
    self.logvar = logvar
    self.fast_sampling = fast_sampling

  def _sample_negatives(self, xs: Tensor, ys: Tensor):
    size = xs.shape[0]
    if self.fast_sampling:
      random_index = torch.randint(size, (size,), device=xs.device).long()
    else:
      random_index = torch.randperm(size)
    return ys[random_index]

  def log_likelihood(self, xs: Tensor, ys: Tensor):
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1)

    logits = self.logvar(xs)
    return -F.cross_entropy(logits, ys)

  def forward(self, xs: Tensor, ys: Tensor, ns: Optional[Tensor] = None, n_negs=1):
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1)
    if ns is not None: ns = ns.reshape(-1)

    logits = self.logvar(xs)

    pos = -F.cross_entropy(logits, ys, reduction='none')
    negs = []

    if ns is not None: assert n_negs == 1, "not implemented"
    for _ in range(n_negs):
      # それぞれの x に対してランダムに一つの y を選ぶ
      if ns is None: ns_nth = self._sample_negatives(xs, ys)
      else: ns_nth = ns
      negs.append(-F.cross_entropy(logits, ns_nth, reduction='none'))

    neg = torch.stack(negs, dim=-1).mean(dim=-1)

    # 公式実装の CLUB (サンプリングしない方) も neg から対角成分を除去する処理は含まず (pos - neg) / 2 を返す一方で、
    # 公式実装の CLUBSample は (pos - neg) / 2 を返していた。
    # 公式実装の CLUBForCategorical では log_mat.diag().mean() - log_mat.mean() を返していたけど半分にしてなかった。
    # だからこの実装でもそうすることにした。正当性は検証してないし、論文も読んでないけど大丈夫でしょ！
    return (pos - neg).mean()

  def learning_loss(self, xs: Tensor, ys: Tensor):
    return -self.log_likelihood(xs, ys)

class CLUBSampleForCategorical3(CLUBSampleForCategorical):
  def log_likelihood(self, xs: Tensor, ys: Tensor):
    logits = self.logvar(xs)

    logits = logits.reshape(-1, logits.shape[-1])
    ys = ys.reshape(-1)

    return -F.cross_entropy(logits, ys)

  def forward(self, xs: Tensor, ys: Tensor, ns: Optional[Tensor] = None, n_negs=1):
    logits = self.logvar(xs)

    logits = logits.reshape(-1, logits.shape[-1])
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1)
    if ns is not None: ns = ns.reshape(-1)

    pos = -F.cross_entropy(logits, ys, reduction='none')
    negs = []

    if ns is not None: assert n_negs == 1, "not implemented"
    for _ in range(n_negs):
      # それぞれの x に対してランダムに一つの y を選ぶ
      if ns is None: ns_nth = self._sample_negatives(xs, ys)
      else: ns_nth = ns
      negs.append(-F.cross_entropy(logits, ns_nth, reduction='none'))

    neg = torch.stack(negs, dim=-1).mean(dim=-1)

    return (pos - neg).mean()

class CLUBSampleForCategorical2(CLUBSampleForCategorical):
  """ 多分、全てのカテゴリが同じ割合で出現するときだけちゃんと機能する """
  def _sample_negatives(self, xs: Tensor, ys: Tensor):
    return torch.randint(0, self.ynum, size=ys.shape, device=xs.device)

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
