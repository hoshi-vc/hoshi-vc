# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

import torch
from torch import nn

from engine.lib.attention import MultiHeadAttention
from engine.lib.layers import ResDown, ResUp, UpSample
from engine.prev.attempt10_dataset import Entry09

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    kdim = 48 * 8
    vdim = 48 * 8

    self.kdim = kdim
    self.vdim = vdim

    base = 48

    # (batch, 1, 80, src_len) -> (batch, kdim, 1, src_len/2)
    self.encode_key = nn.Sequential(
        nn.Conv2d(1, base, kernel_size=3, padding=1),
        ResDown(base * 1, base * 2, normalize=True, downsample="half"),
        ResDown(base * 2, base * 4, normalize=True, downsample="time-preserve"),
        ResDown(base * 4, base * 8, normalize=True, downsample="time-preserve"),
        ResDown(base * 8, base * 8, normalize=True, downsample="time-preserve"),
        ResDown(base * 8, base * 8, normalize=True),
        ResDown(base * 8, kdim, normalize=True),
        nn.AvgPool2d((5, 1)),
    )

    # (batch, 1, 80, src_len) -> (batch, vdim, 1, src_len/2)
    self.encode_value = nn.Sequential(
        nn.Conv2d(1, base, kernel_size=3, padding=1),
        ResDown(base * 1, base * 2, normalize=True, downsample="half"),
        ResDown(base * 2, base * 4, normalize=True, downsample="time-preserve"),
        ResDown(base * 4, base * 8, normalize=True, downsample="time-preserve"),
        ResDown(base * 8, base * 8, normalize=True, downsample="time-preserve"),
        ResDown(base * 8, base * 8, normalize=True),
        ResDown(base * 8, vdim, normalize=True),
        nn.AvgPool2d((5, 1)),
    )

    self.lookup = MultiHeadAttention(kdim, vdim, 16, dropout=0.2, hard=True)

    # (batch, vdim, 1, src_len/2) -> (batch, 1, 80, src_len)
    self.decode = nn.Sequential(
        UpSample("time-preserve", 5),
        ResUp(vdim, base * 2, normalize=True),
        ResUp(base * 2, base * 2, normalize=True),
        ResUp(base * 2, base * 2, normalize=True, upsample="time-preserve"),
        ResUp(base * 2, base * 2, normalize=True, upsample="time-preserve"),
        ResUp(base * 2, base * 2, normalize=True, upsample="time-preserve"),
        ResUp(base * 2, base * 1, normalize=True, upsample="half"),
        nn.Conv2d(base, 1, kernel_size=1, padding=0),
    )

  def forward(self, batch: Entry09):
    n_refs = len(batch.ref)
    n_batch = len(batch.src.mel)
    ref_len = batch.ref[0].mel.shape[1]

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_mel = torch.stack([o.mel for o in batch.ref]).flatten(0, 1)

    src_mel = batch.src.mel

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_key = self.encode_key(ref_mel.unsqueeze(1).transpose(2, 3)).squeeze(2).transpose(1, 2)
    ref_value = self.encode_value(ref_mel.unsqueeze(1).transpose(2, 3)).squeeze(2).transpose(1, 2)

    src_key = self.encode_key(src_mel.unsqueeze(1).transpose(2, 3)).squeeze(2).transpose(1, 2)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim) -> (n_batch, n_refs*seq_len, feat_dim)
    ref_key = ref_key.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_value = ref_value.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)

    assert ref_key.shape[1] == (ref_len / 2) * n_refs, f"ref_key.shape={ref_key.shape}, ref_len={ref_len}, n_refs={n_refs}"

    tgt_value, attn = self.lookup(src_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(tgt_value.transpose(1, 2).unsqueeze(2)).squeeze(1).transpose(1, 2)

    return tgt_mel, (ref_key, ref_value, None, attn), []
