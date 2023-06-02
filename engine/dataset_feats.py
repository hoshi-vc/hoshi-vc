from random import Random
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from engine.preparation import CREPE_MODEL, PITCH_TOPK

class FeatureEntry(NamedTuple):
  mel: Tensor
  w2v2: Tensor
  pitch_i: Tensor
  pitch_v: Tensor

class FeatureDataset(Dataset):
  def __init__(self, dirs: list[str], frames: int, start_hop: int):
    self.random = Random()
    self.dirs = dirs
    self.frames = frames
    self.start_hop = start_hop

    self.starts: list[tuple[str, int]] = []
    for d in dirs:
      length = np.load(d / "mel.npy", mmap_mode="r").shape[0]
      for i in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, i))

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> FeatureEntry:
    d, start = self.starts[index]
    start += self.random.randint(0, self.start_hop)
    end = start + self.frames

    return FeatureEntry(
        mel=torch.as_tensor(np.array(np.load(d / "mel.npy", mmap_mode="r")[start:end])),
        w2v2=torch.as_tensor(np.array(np.load(d / "w2v2.npy", mmap_mode="r")[start:end])),
        pitch_i=torch.as_tensor(np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end])),
        pitch_v=torch.as_tensor(np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end])),
    )
