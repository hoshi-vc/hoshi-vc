# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from random import Random
from typing import NamedTuple, Optional

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, Subset

from engine.dataset_feats import IntraDomainDataModule, IntraDomainDataset
from engine.singleton import (CREPE_MODEL, FEATS_DIR, PHONEME_TOPK, PITCH_TOPK, Preparation)

# TODO: dataset_feats と重複が多いので、整理する

class FeatureEntryA07(NamedTuple):
  audio: Tensor
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  soft: Tensor

class IntraDomainEntryA07(NamedTuple):
  src: FeatureEntryA07
  aug: FeatureEntryA07
  ref: list[FeatureEntryA07]

class IntraDomainDatasetA07(IntraDomainDataset):
  """ ランダムサンプリングされた音声 + その同一話者の複数の音声 (公式実装と同じく n_samples + 1 個の要素を返す) """
  def __init__(self,
               src_dirs: list[str],
               aug_dirs: list[str],
               speaker_ids: list[int],
               frames: int,
               start_hop: int,
               n_samples: int,
               shuffle: Optional[int] = None):
    super().__init__(dirs=aug_dirs, speaker_ids=speaker_ids, frames=frames, start_hop=start_hop, n_samples=n_samples, shuffle=shuffle, random_offset=False)

    assert len(aug_dirs) == len(src_dirs)
    self.src_dirs = src_dirs
    self.src_lut = {aug: src for src, aug in zip(src_dirs, aug_dirs)}

  def load_entry(self, d: str, speaker_id: int, start: int, frames: int, aug=False) -> FeatureEntryA07:
    end = start + frames

    if not aug: d = self.src_lut[d]

    return FeatureEntryA07(
        audio=np.array(np.load(d / "audio.npy", mmap_mode="r")[start * 256:end * 256]),
        speaker=np.array([speaker_id]),
        energy=np.array(np.load(d / "energy.npy", mmap_mode="r")[start:end]),
        mel=np.array(np.load(d / "melspec.npy", mmap_mode="r")[start:end]),
        pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
        soft=np.array(np.load(d / "hubert_soft.npy", mmap_mode="r")[start:end]),
    )

  def __getitem__(self, index: int):
    d, speaker_id, start = self.starts[index]

    other_indices = self.same_speaker_lut[speaker_id]
    other_indices = self.random2.sample(other_indices, self.n_samples + 1)

    src = self.load_entry(d, speaker_id, start, self.frames)
    aug = self.load_entry(d, speaker_id, start, self.frames, aug=True)

    ref: list[FeatureEntryA07] = []
    for i in other_indices:
      if i == index: continue
      if len(ref) == self.n_samples: break
      d, sid2, start = self.starts[i]
      assert speaker_id == sid2
      ref.append(self.load_entry(d, speaker_id, start, self.frames))

    return IntraDomainEntryA07(src, aug, ref)

class IntraDomainDataModuleA07(IntraDomainDataModule):
  def __init__(self, P: Preparation, aug_root: str, frames: int, n_samples: int, batch_size: int, num_workers=4, n_batches=None, n_batches_val=None):
    super().__init__(P, frames, n_samples, batch_size, num_workers, dataset_class=IntraDomainDatasetA07)
    self.aug_root = aug_root
    self.n_batches = n_batches
    self.n_batches_val = n_batches_val

  def setup(self, stage: str):
    self.P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    train_augs = [self.aug_root / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    valid_augs = [self.aug_root / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    speaker_ids = [i for i, _ in enumerate(self.P.dataset.speaker_ids)]

    self.intra_train = self.dataset_class(train_dirs, train_augs, speaker_ids, self.frames, self.frames, self.n_samples)
    self.intra_valid = self.dataset_class(valid_dirs, valid_augs, speaker_ids, self.frames, self.frames, self.n_samples, shuffle=7892639)
    self.intra_valid = Subset(self.intra_valid, Random(37892834).choices(list(range(len(self.intra_valid))), k=self.n_batches_val * self.batch_size))

  def train_dataloader(self):
    if self.n_batches is None: return super().train_dataloader()
    return DataLoader(
        self.intra_train,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        sampler=RandomSampler(self.intra_train, replacement=True, num_samples=self.n_batches * self.batch_size),
        drop_last=True,
        pin_memory=True,
    )
