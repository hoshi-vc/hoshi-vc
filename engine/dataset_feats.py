# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from random import Random
from typing import Any, Callable, NamedTuple, Optional

import lightning.pytorch as L
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from engine.prepare import (CREPE_MODEL, FEATS_DIR, PHONEME_TOPK, PITCH_TOPK, Preparation)

EntryLoader = Callable[[str, int, int], Any]

class FeatureEntry(NamedTuple):
  mel: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  w2v2: Tensor

class FeatureDataset(Dataset):
  def __init__(self, dirs: list[str], frames: int, start_hop: int, random_offset: bool):
    dirs = sorted(dirs)

    self.random = Random(43248650)
    self.dirs = dirs
    self.frames = frames
    self.start_hop = start_hop
    self.random_offset = random_offset

    self.starts: list[tuple[str, int]] = []
    for d in dirs:
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      for i in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, i))

  def load_entry(self, d: str, start: int, frames: int) -> FeatureEntry:
    return load_feature_entry(d, start, frames)

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
  def __init__(self, dirs: list[str], speaker_ids: list[int], frames: int, start_hop: int, n_samples: int, random_offset: bool, shuffle: Optional[int] = None):

    assert len(dirs) == len(speaker_ids)
    self.random = Random(35534253)
    self.random2 = Random(54235235)
    self.frames = frames
    self.start_hop = start_hop
    self.n_samples = n_samples
    self.random_offset = random_offset

    self.starts: list[tuple[str, int, int]] = []
    for d, speaker_id in zip(dirs, speaker_ids):
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      for start in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, speaker_id, start))

    if shuffle is not None:
      rand = Random(shuffle)
      rand.shuffle(self.starts)

    # シャッフルの前に same_speaker_lut を作成するというバグがあった。
    # なので、この修正を施す前のすべての実験結果は無効になります！やり直しだぁ！
    self.same_speaker_lut: dict[str, list[int]] = {}
    for i, (d, speaker_id, start) in enumerate(self.starts):
      self.same_speaker_lut.setdefault(speaker_id, []).append(i)

    for k, v in self.same_speaker_lut.items():
      if len(v) < n_samples + 1:
        raise ValueError(f"Speaker {k} has only {len(v)} samples, which is less than {n_samples} + 1.")

  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> FeatureEntry:
    return load_feature_entry(d, start, frames)

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> IntraDomainEntry:
    d, speaker_id, start = self.starts[index]
    offset = 0
    if self.random_offset: self.random.randint(0, self.start_hop)

    other_indices = self.same_speaker_lut[speaker_id]
    other_indices = self.random2.sample(other_indices, self.n_samples + 1)

    entries: list[FeatureEntry] = []
    entries.append(self.load_entry(d, speaker_id, start + offset, self.frames))
    for i in other_indices:
      if i == index: continue
      if len(entries) == self.n_samples + 1: break
      d, sid2, start = self.starts[i]
      assert speaker_id == sid2  # 今後は same_speaker_lut が壊れてたらここでエラーになるはず
      entries.append(self.load_entry(d, speaker_id, start + offset, self.frames))

    return entries

class IntraDomainDataModule(L.LightningDataModule):
  # 公式実装では intra_valid からも intra_train の音声を参照用にサンプリングしてる。
  # TODO: うまくいかなかったら、公式実装と同じ挙動にしてみる。
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4, dataset_class=IntraDomainDataset):
    super().__init__()
    self.P = P
    self.frames = frames
    self.n_samples = n_samples
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.dataset_class = dataset_class
    self.intra_train: Any = None
    self.intra_valid: Any = None

  def setup(self, stage: str):
    self.P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    speaker_ids = [i for i, _ in enumerate(self.P.dataset.speaker_ids)]

    # TODO: club の計算時にはバッチ内の多様性が大切なので、 valid_dataset も固定シードでシャッフルすることにした
    self.intra_train = self.dataset_class(train_dirs, speaker_ids, self.frames, self.frames, self.n_samples, random_offset=True)
    self.intra_valid = self.dataset_class(valid_dirs, speaker_ids, self.frames, self.frames, self.n_samples, random_offset=False, shuffle=7892639)

  def train_dataloader(self):
    return DataLoader(
        self.intra_train,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

  def val_dataloader(self):
    return DataLoader(
        self.intra_valid,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

def load_feature_entry(d: str, start: int, frames: int) -> FeatureEntry:
  end = start + frames

  return FeatureEntry(
      mel=np.array(np.load(d / "melspec.npy", mmap_mode="r")[start:end]),
      pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
      pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
      w2v2=np.array(np.load(d / "wav2vec2.npy", mmap_mode="r")[start:end]),
  )

class FeatureEntry2(NamedTuple):
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  phoneme_i: Tensor
  phoneme_v: Tensor
  pitch_i: Tensor
  pitch_v: Tensor

IntraDomainEntry2 = list[FeatureEntry2]

class IntraDomainDataset2(IntraDomainDataset):
  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> FeatureEntry2:
    end = start + frames

    return FeatureEntry2(
        speaker=np.array([speaker_id]),
        energy=np.array(np.load(d / "energy.npy", mmap_mode="r")[start:end]),
        mel=np.array(np.load(d / "melspec.npy", mmap_mode="r")[start:end]),
        phoneme_i=np.array(np.load(d / f"phoneme_i_{PHONEME_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        phoneme_v=np.array(np.load(d / f"phoneme_v_{PHONEME_TOPK}.npy", mmap_mode="r")[start:end]),
        pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
    )

  def __getitem__(self, index: int) -> IntraDomainEntry2:
    return super().__getitem__(index)

class IntraDomainDataModule2(IntraDomainDataModule):
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4):
    super().__init__(P, frames, n_samples, batch_size, num_workers, dataset_class=IntraDomainDataset2)

class FeatureEntry3(NamedTuple):
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  w2v2: Tensor

IntraDomainEntry3 = list[FeatureEntry3]

class IntraDomainDataset3(IntraDomainDataset):
  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> FeatureEntry3:
    end = start + frames

    return FeatureEntry3(
        speaker=np.array([speaker_id]),
        energy=np.array(np.load(d / "energy.npy", mmap_mode="r")[start:end]),
        mel=np.array(np.load(d / "melspec.npy", mmap_mode="r")[start:end]),
        pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
        w2v2=np.array(np.load(d / "wav2vec2.npy", mmap_mode="r")[start:end]),
    )

  def __getitem__(self, index: int) -> FeatureEntry3:
    return super().__getitem__(index)

class IntraDomainDataModule3(IntraDomainDataModule):
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4):
    super().__init__(P, frames, n_samples, batch_size, num_workers, dataset_class=IntraDomainDataset3)

class FeatureEntry4(NamedTuple):
  audio: Tensor
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  phoneme_i: Tensor
  phoneme_v: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  soft: Tensor
  w2v2: Tensor

IntraDomainEntry4 = list[FeatureEntry4]

class IntraDomainDataset4(IntraDomainDataset):
  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> FeatureEntry4:
    end = start + frames

    return FeatureEntry4(
        audio=np.array(np.load(d / "audio.npy", mmap_mode="r")[start * 256:end * 256]),
        speaker=np.array([speaker_id]),
        energy=np.array(np.load(d / "energy.npy", mmap_mode="r")[start:end]),
        mel=np.array(np.load(d / "melspec.npy", mmap_mode="r")[start:end]),
        phoneme_i=np.array(np.load(d / f"phoneme_i_{PHONEME_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        phoneme_v=np.array(np.load(d / f"phoneme_v_{PHONEME_TOPK}.npy", mmap_mode="r")[start:end]),
        pitch_i=np.array(np.load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end], np.int64),
        pitch_v=np.array(np.load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy", mmap_mode="r")[start:end]),
        soft=np.array(np.load(d / "hubert_soft.npy", mmap_mode="r")[start:end]),
        w2v2=np.array(np.load(d / "wav2vec2.npy", mmap_mode="r")[start:end]),
    )

  def __getitem__(self, index: int) -> FeatureEntry4:
    return super().__getitem__(index)

class IntraDomainDataModule4(IntraDomainDataModule):
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4, n_batches=None, n_batches_val=None):
    super().__init__(P, frames, n_samples, batch_size, num_workers, dataset_class=IntraDomainDataset4)
    self.n_batches = n_batches
    self.n_batches_val = n_batches_val

  def setup(self, stage: str):
    super().setup(stage)
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
