# %%

from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.functional as F
import torch.nn.functional as F
from torch import Tensor, nn

from engine.lib.layers import Buckets, GetNth, Transpose

class Input04(NamedTuple):
  src_energy: Tensor  #    (batch, src_len, 1)
  src_phoneme_i: Tensor  # (batch, src_len, topk)
  src_phoneme_v: Tensor  # (batch, src_len, topk)
  src_pitch_i: Tensor  #   (batch, src_len, 1+)
  ref_energy: Tensor  #    (batch, ref_len, 1)
  ref_phoneme_i: Tensor  # (batch, ref_len, topk)
  ref_phoneme_v: Tensor  # (batch, src_len, topk)
  ref_pitch_i: Tensor  #   (batch, ref_len, 1+)
  ref_mel: Tensor  #       (batch, ref_len, 80)

class Model04(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: normalization, dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    others_dim = hdim - energy_dim - pitch_dim
    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.phoneme_embed = nn.Embedding(400, others_dim)
    self.encode_key = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.RNN(hdim, hdim, batch_first=True),
        GetNth(0),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
    )

    self.mel_encode = nn.Linear(80, others_dim)
    self.encode_value = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
    )

    self.lookup = nn.MultiheadAttention(hdim, 4, dropout=0.1, batch_first=True)

    self.decode = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, 80, kernel_size=1),
        Transpose(1, 2),
    )

  def forward_key(self, energy_i: Tensor, pitch_i: Tensor, phoneme_i: Tensor, phoneme_v: Tensor):
    energy = self.energy_embed(self.energy_bins(energy_i[:, :, 0]))
    pitch = self.pitch_embed(pitch_i[:, :, 0])

    phoneme: Optional[Tensor] = None
    for k in range(phoneme_v.shape[-1]):
      emb_k = self.phoneme_embed(phoneme_i[:, :, k])
      emb_k *= phoneme_v[:, :, k].exp().unsqueeze(-1)
      phoneme = emb_k if phoneme is None else phoneme + emb_k

    key = torch.cat([energy, pitch, phoneme], dim=-1)
    key = self.encode_key(key)

    return key

  def forward_value(self, energy_i: Tensor, pitch_i: Tensor, mel: Tensor):
    energy = self.energy_embed(self.energy_bins(energy_i[:, :, 0]))
    pitch = self.pitch_embed(pitch_i[:, :, 0])
    mel = self.mel_encode(mel)

    value = torch.cat([energy, pitch, mel], dim=-1)
    value = self.encode_value(value)

    return value

  def forward(self, o: Input04):
    # shape: (batch, src_len, hdim)
    src_key = self.forward_key(o.src_energy, o.src_pitch_i, o.src_phoneme_i, o.src_phoneme_v)
    ref_key = self.forward_key(o.ref_energy, o.ref_pitch_i, o.ref_phoneme_i, o.ref_phoneme_v)
    ref_value = self.forward_value(o.ref_energy, o.ref_pitch_i, o.ref_mel)

    tgt_value, _ = self.lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(tgt_value)

    return tgt_mel, (ref_key, ref_value)
