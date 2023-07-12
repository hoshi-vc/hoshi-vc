# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

import torch
from torch import Tensor, nn

from engine.lib.layers import Buckets, Transpose
from engine.prev.attempt10_dataset import Entry09

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim // 2
    mel_dim = hdim // 2
    kv_dim = hdim // 2

    self.kv_dim = kv_dim

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.w2v2_embed = nn.Linear(256, w2v2_dim)
    self.encode_key = nn.Sequential(
        nn.Linear(energy_dim + pitch_dim + w2v2_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.mel_encode = nn.Linear(80, mel_dim)
    self.encode_value = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.lookup1 = nn.MultiheadAttention(kv_dim, 16, dropout=0.2, batch_first=True)

    self.joint1 = nn.Sequential(
        nn.Linear(kv_dim + energy_dim + pitch_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.encode_value2 = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.lookup2 = nn.MultiheadAttention(kv_dim, 16, dropout=0.2, batch_first=True)

    self.decode = nn.Sequential(
        nn.Linear(kv_dim + energy_dim + pitch_dim, hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        # PosFFT(hdim, layers=2, heads=2, hdim=256, kernels=(3, 3), dropout=0.2, posenc_len=2048),
        nn.Linear(hdim, 80),
    )

  def forward_energy(self, energy_i: Tensor):
    return self.energy_embed(self.energy_bins(energy_i[:, :, 0]))

  def forward_pitch(self, pitch_i: Tensor):
    return self.pitch_embed(pitch_i[:, :, 0])

  def forward_w2v2(self, w2v2: Tensor):
    return self.w2v2_embed(w2v2)

  def forward_mel(self, mel: Tensor):
    return self.mel_encode(mel)

  def forward_key(self, energy: Tensor, pitch: Tensor, w2v2: Tensor):
    return self.encode_key(torch.cat([energy, pitch, w2v2], dim=-1))

  def forward_value1(self, energy: Tensor, pitch: Tensor, mel: Tensor):
    return self.encode_value(torch.cat([energy, mel], dim=-1))

  def forward_value2(self, energy: Tensor, pitch: Tensor, mel: Tensor):
    return self.encode_value2(torch.cat([energy, mel], dim=-1))

  def forward(self, batch: Entry09):
    n_refs = len(batch.ref)
    n_batch = len(batch.src.energy)
    ref_len = batch.ref[0].energy.shape[1]

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_energy = torch.stack([o.energy for o in batch.ref]).flatten(0, 1)
    ref_pitch_i = torch.stack([o.pitch_i for o in batch.ref]).flatten(0, 1)
    ref_w2v2 = torch.stack([o.soft for o in batch.ref]).flatten(0, 1)
    ref_mel = torch.stack([o.mel for o in batch.ref]).flatten(0, 1)

    src_energy = batch.src.energy
    src_pitch_i = batch.src.pitch_i
    src_w2v2 = batch.src.soft
    src_mel = batch.src.mel

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_energy = self.forward_energy(ref_energy)
    ref_pitch = self.forward_pitch(ref_pitch_i)
    ref_w2v2 = self.forward_w2v2(ref_w2v2)
    ref_mel = self.forward_mel(ref_mel)
    ref_key = self.forward_key(ref_energy, ref_pitch, ref_w2v2)
    ref_value1 = self.forward_value1(ref_energy, ref_pitch, ref_mel)
    ref_value2 = self.forward_value2(ref_energy, ref_pitch, ref_mel)

    src_energy = self.forward_energy(src_energy)
    src_pitch = self.forward_pitch(src_pitch_i)
    src_w2v2 = self.forward_w2v2(src_w2v2)
    src_mel = self.forward_mel(src_mel)
    src_key = self.forward_key(src_energy, src_pitch, src_w2v2)
    # src_value = self.forward_value(src_energy, src_pitch, src_mel)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim) -> (n_batch, n_refs*seq_len, feat_dim)
    ref_energy = ref_energy.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_pitch = ref_pitch.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_key = ref_key.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_value1 = ref_value1.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_value2 = ref_value2.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_pitch_i = ref_pitch_i.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)

    assert ref_key.shape[1] == ref_len * n_refs, f"ref_key.shape={ref_key.shape}, ref_len={ref_len}, n_refs={n_refs}"

    tgt_value1, attn1 = self.lookup1(src_key, ref_key, ref_value1, need_weights=True)
    tgt_key1 = self.joint1(torch.cat([tgt_value1, src_energy, src_pitch], dim=-1))
    tgt_value2, attn2 = self.lookup1(tgt_key1, ref_key, ref_value2, need_weights=True)

    tgt_value = tgt_value1 + tgt_value2  # residual connection

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    return tgt_mel, (ref_key, ref_value1, ref_pitch_i, (attn1, attn2)), []
