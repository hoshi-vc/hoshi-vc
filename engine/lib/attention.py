# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

F.multi_head_attention_forward

def sdp_attn(Q, K, V, dropout=0.0, need_weights=False) -> Tensor:
  """ https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html """

  # if not need_weights: return F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout)

  attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
  if dropout > 0.0: attn_weight = torch.dropout(attn_weight, dropout, True)
  return attn_weight @ V, attn_weight

def sdp_attn_discrete(Q, K, V, dropout=0.0, need_weights=False) -> Tensor:
  # attn_weight: (n*b) x lq x lk
  attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
  if dropout > 0.0: attn_weight = torch.dropout(attn_weight, dropout, True)

  # continous: (n*b) x lq x dv
  continous = attn_weight @ V

  discrete_i = attn_weight.argmax(dim=-1, keepdim=True)
  # discrete[i, j, k] = V[i, discrete_i[i, j, k], k]
  discrete = torch.gather(V, dim=1, index=discrete_i.expand(-1, -1, V.size(-1)))

  assert continous.shape == discrete.shape
  return discrete.detach() + continous - continous.detach(), attn_weight

class MultiHeadAttention(nn.Module):
  def __init__(self, d_k: int, d_v: int, n_heads: int, dropout: float, hard=False):
    super().__init__()

    self.n_heads = n_heads
    self.kdim = d_k
    self.vdim = d_v
    self.dropout = dropout
    self.hard = hard

    self.w_qs = nn.Linear(d_k, n_heads * d_k)
    self.w_ks = nn.Linear(d_k, n_heads * d_k)
    self.w_vs = nn.Linear(d_v, n_heads * d_v)

    self.layer_norm = nn.LayerNorm(d_v)

    self.fc = nn.Linear(n_heads * d_v, d_v)

  def forward(self, q: Tensor, k: Tensor, v: Tensor, need_weights=False):
    kdim, vdim, n_head = self.kdim, self.vdim, self.n_heads

    sz_b, len_q, _ = q.shape
    sz_b, len_k, _ = k.shape
    sz_b, len_v, _ = v.shape

    q = self.w_qs(q).view(sz_b, len_q, n_head, kdim)
    k = self.w_ks(k).view(sz_b, len_k, n_head, kdim)
    v = self.w_vs(v).view(sz_b, len_v, n_head, vdim)
    q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, kdim)  # (n*b) x lq x dk
    k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, kdim)  # (n*b) x lk x dk
    v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, vdim)  # (n*b) x lv x dv

    dropout = self.dropout if self.training else 0.0

    if not self.hard: output, attn = sdp_attn(q, k, v, dropout, need_weights)
    else: output, attn = sdp_attn_discrete(q, k, v, dropout, need_weights)

    if need_weights: attn = attn.view(n_head, sz_b, len_q, len_k).mean(dim=0)

    output = output.view(n_head, sz_b, len_q, vdim)
    output = (output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1))  # b x lq x (n*dv)

    output = self.fc(output)

    return output, attn
