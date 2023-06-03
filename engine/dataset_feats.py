from random import Random
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from engine.preparation import CREPE_MODEL, PITCH_TOPK

EntryLoader = Callable[[str, int, int], Any]

class FeatureEntry(NamedTuple):
  mel: Tensor
  w2v2: Tensor
  pitch_i: Tensor
  pitch_v: Tensor

class FeatureDataset(Dataset):
  def __init__(self, dirs: list[str], frames: int, start_hop: int, random_offset: bool, loader: Optional[EntryLoader] = None):
    self.random = Random()
    self.dirs = dirs
    self.frames = frames
    self.start_hop = start_hop
    self.random_offset = random_offset
    self.entry_loader = loader or load_feature_entry

    self.starts: list[tuple[str, int]] = []
    for d in dirs:
      length = np.load(d / "mel.npy", mmap_mode="r").shape[0]
      for i in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, i))

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> FeatureEntry:
    d, start = self.starts[index]
    offset = 0
    if self.random_offset: offset = self.random.randint(0, self.start_hop)
    start += offset
    return self.entry_loader(d, start, self.frames)

IntraDomainEntry = list[FeatureEntry]

class IntraDomainDataset(Dataset):
  """ ランダムサンプリングされた音声 + その同一話者の複数の音声 (公式実装と同じく n_samples + 1 個の要素を返す) """
  def __init__(self, dirs: list[str], frames: int, start_hop: int, n_samples: int, random_offset: bool, loader: Optional[EntryLoader] = None):
    self.random = Random()
    self.random2 = Random()
    self.dirs = dirs
    self.frames = frames
    self.start_hop = start_hop
    self.n_samples = n_samples
    self.random_offset = random_offset
    self.entry_loader = loader or load_feature_entry

    self.starts: list[tuple[str, int]] = []
    self.same_domain_lut: dict[str, list[int]] = {}
    for d in dirs:
      length = np.load(d / "mel.npy", mmap_mode="r").shape[0]
      for i in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, i))
        self.same_domain_lut.setdefault(d, []).append(len(self.starts) - 1)

    for k, v in self.same_domain_lut.items():
      if len(v) < n_samples + 1:
        raise ValueError(f"Domain {k} has only {len(v)} samples, which is less than {n_samples} + 1.")

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> IntraDomainEntry:
    d, start = self.starts[index]
    offset = 0
    if self.random_offset: self.random.randint(0, self.start_hop)

    other_indices = self.same_domain_lut[d]
    other_indices = self.random2.sample(other_indices, self.n_samples + 1)

    entries: list[FeatureEntry] = []
    entries.append(self.entry_loader(d, start + offset, self.frames))
    for i in other_indices:
      if i == index: continue
      if len(entries) == self.n_samples + 1: break
      d, start = self.starts[i]
      entries.append(self.entry_loader(d, start + offset, self.frames))

    return entries

def load_feature_entry(d: str, start: int, frames: int) -> FeatureEntry:
  end = start + frames

  return FeatureEntry(
      mel=np.array(np.load(d / "mel.npy", mmap_mode="r")[start:end]),
      w2v2=np.array(np.load(d / "w2v2.npy", mmap_mode="r")[start:end]),
      pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
      pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
  )
