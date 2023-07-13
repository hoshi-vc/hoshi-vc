# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from os import sep
from pathlib import Path
from random import Random
from typing import Any, NamedTuple, Optional

import faiss
import lightning.pytorch as L
import numpy as np
from faiss import IndexPreTransform
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from engine.attempts.a10_prepare import LUT_ROOT
from engine.lib.utils import NPArray, clamp
from engine.singleton import (CREPE_MODEL, FEATS_DIR, PHONEME_TOPK, PITCH_TOPK, P)

@dataclass
class FeatsList:
  audio: bool = False
  speaker: bool = False
  energy: bool = False
  mel: bool = False
  phoneme_i: bool = False
  phoneme_v: bool = False
  pitch_i: bool = False
  pitch_v: bool = False
  soft: bool = False

  @staticmethod
  def all():
    return FeatsList(True, True, True, True, True, True, True, True, True)

class Feats10(NamedTuple):
  audio: Tensor
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  phoneme_i: Tensor
  phoneme_v: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  soft: Tensor

  def load(accessor: Optional["NPAccessor"], d: Path | str, speaker_id: int, start: int, frames: int, req: FeatsList | None = None) -> "Feats10":
    end = start + frames

    if req is None: req = FeatsList.all()

    # Path を使うときのオーバーヘッドが無視できないので、 sep で単純につなげる
    d = str(d)
    if accessor is None: load = lambda p: np.load(d + sep + p, mmap_mode="r")
    else: load = lambda p: accessor.load(d + sep + p)

    return Feats10(
        audio=np.array(load("audio.npy")[start * 256:end * 256]) if req.audio else (),
        speaker=np.array([speaker_id]) if req.speaker else (),
        energy=np.array(load("energy.npy")[start:end]) if req.energy else (),
        mel=np.array(load("melspec.npy")[start:end]) if req.mel else (),
        phoneme_i=np.array(load(f"phoneme_i_{PHONEME_TOPK}.npy")[start:end], np.int64) if req.phoneme_i else (),
        phoneme_v=np.array(load(f"phoneme_v_{PHONEME_TOPK}.npy")[start:end]) if req.phoneme_v else (),
        pitch_i=np.array(load(f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy")[start:end], np.int64) if req.pitch_i else (),
        pitch_v=np.array(load(f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy")[start:end]) if req.pitch_v else (),
        soft=np.array(load("hubert_soft.npy")[start:end]) if req.soft else (),
    )

class Entry10(NamedTuple):
  src: Feats10
  ref: list[Feats10]

class NPAccessor:
  """ open_memmap が遅いので、キャッシュする """
  def __init__(self, cache: bool):
    self.cache = cache
    self.cache_dict: dict[str, NPArray] = {}

  def load(self, p: str) -> NPArray:
    if p in self.cache_dict: return self.cache_dict[p]
    if self.cache:
      self.cache_dict[p] = np.load(p, mmap_mode="r")

      # 各プロセスごとに独立した cache_dict が作られてた
      # print(getpid(), len(self.cache_dict))

      return self.cache_dict[p]
    return np.load(p, mmap_mode="r")

  def clear_cache(self) -> None:
    self.cache_dict.clear()

class FeatsDataset10(Dataset):
  """ Feats10 のデータセット """
  def __init__(
      self,
      accessor: NPAccessor,
      dirs: list[str],
      speaker_ids: list[int],
      frames: int,
      drop_last: bool,
      req: FeatsList | None = None,
  ):
    assert len(dirs) == len(speaker_ids)

    self.accessor = accessor
    self.frames = frames
    self.req = req

    self.starts: list[tuple[str, int, int]] = []
    for d, speaker_id in zip(dirs, speaker_ids):
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      start_last = length - frames if drop_last else length
      for start in range(0, start_last, frames):
        self.starts.append((d, speaker_id, start))

  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> Feats10:
    return Feats10.load(self.accessor, d, speaker_id, start, frames, self.req)

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> Entry10:
    d, speaker_id, start = self.starts[index]

    return self.load_entry(d, speaker_id, start, self.frames)

class Dataset10(Dataset):
  """ Reference をいい感じに選ぶデータセット """
  def __init__(
      self,
      accessor: NPAccessor,
      dirs: list[str],
      dirs_ref: list[str],
      indices: list[IndexPreTransform],
      speaker_ids: list[int],
      frames: int,
      frames_ref: int,
      n_refs: int,
      ref_max_kth: int,
      rand_refs: int,
      rand_ref_kth: int,
      rand_start: Optional[int],
      shuffle: Optional[int],
      same_density=False,
      req: FeatsList | None = None,
  ):
    assert len(dirs) == len(indices) == len(speaker_ids)
    assert frames >= frames_ref

    self.accessor = accessor
    self.rand_start = rand_start and Random(rand_start)
    self.rand_refs = rand_refs and Random(rand_refs)
    self.rand_ref_kth = rand_ref_kth and np.random.default_rng(rand_ref_kth)
    self.frames = frames
    self.frames_ref = frames_ref
    self.n_refs = n_refs
    self.ref_max_kth = ref_max_kth
    self.same_density = same_density
    self.req = req

    self.starts: list[tuple[str, str, IndexPreTransform, int, int]] = []
    self.max_ref_lens: dict[str, int] = {}
    for d, dref, faiss, speaker_id in zip(dirs, dirs_ref, indices, speaker_ids):
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      self.max_ref_lens[dref] = np.load(dref / "melspec.npy", mmap_mode="r").shape[0]
      for start in range(0, length - frames * 2, frames):
        self.starts.append((d, dref, faiss, speaker_id, start))

    if shuffle: Random(shuffle).shuffle(self.starts)

  def _load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> Feats10:
    return Feats10.load(self.accessor, d, speaker_id, start, frames, self.req)

  def _get_key(self, d: str, speaker_id: int, start: int, frames: int, feats: Feats10, key_indices: list[int]) -> Tensor:
    return feats.soft[key_indices]

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> Entry10:
    d, dref, faiss, speaker_id, start = self.starts[index]
    if self.rand_start: start += self.rand_start.randint(0, self.frames)

    max_ref_len = self.max_ref_lens[dref]

    src = self._load_entry(d, speaker_id, start, self.frames)

    if not self.same_density:
      key_indices = self.rand_refs.sample(range(len(src.soft)), self.n_refs)
      key_indices.sort()
    else:
      key_indices = np.linspace(0, len(src.soft) - self.frames_ref * 2, self.n_refs, dtype=np.int64)
      key_indices += self.rand_refs.randint(0, self.frames_ref)

    keys = self._get_key(d, speaker_id, start, self.frames, src, key_indices)

    _, ref_indices = faiss.search(keys.astype(np.float32), self.ref_max_kth)

    # データリークを避けて、一番近そうなフラグメントを選ぶ。
    # 計算量的には十分大丈夫だと思う。
    # TODO: 総合的に見て一番網羅的な選び方をするとかしてみたいけど、パワーが足りん。
    ref_starts = []
    for i in range(self.n_refs):
      can_be: list[int] = []
      ref_start: int | None = None
      for k in range(self.ref_max_kth):
        candidate_middle = ref_indices[i, k]
        candidate_start = clamp(candidate_middle - self.frames_ref // 2, 0, max_ref_len - self.frames_ref)

        # 変換すべき音声と重なってたら、次の候補を探す。
        if d == dref and intersect(start, start + self.frames, candidate_start, candidate_start + self.frames_ref): continue

        can_be.append(candidate_start)

        # 他の参照音声と重なってたら、次の候補を探す。
        margin = self.frames_ref // 4
        if any(ref_start + margin <= candidate_middle < ref_start + self.frames_ref - margin for ref_start in ref_starts): continue

        # この参照音声を採用する。
        ref_start = candidate_start
        # if k > 1: print(k)
        break

      # どの候補もダメだったら、一つ前に従う
      if ref_start is None:
        if len(can_be) > 0:
          # print(f"Warn: No suitable reference found. Falling back. (spk={speaker_id}, start={start})")
          ref_start = can_be[0]
        elif i > 0:
          # print(f"Warn: No suitable reference found. Falling back. (spk={speaker_id}, start={start})")
          ref_start = ref_starts[-1]
        else:
          print(f"Error: No suitable reference found. Please increase ref_max_kth. (spk={speaker_id}, start={start})")
          # どうせこのコードパスには来ないので、雑に変換すべき音声の周りを使う
          ref_start = start - self.frames_ref
          if ref_start < 0: ref_start = start + self.frames
          assert not intersect(start, start + self.frames, ref_start, ref_start + self.frames_ref)

      ref_starts.append(ref_start)

    assert len(ref_starts) == self.n_refs

    refs: list[Feats10] = []
    for ref_start in ref_starts:
      ref = self._load_entry(dref, speaker_id, ref_start, self.frames_ref)
      refs.append(ref)

    return Entry10(src, refs)

def intersect(start1: int, end1: int, start2: int, end2: int) -> bool:
  """ [start1, end1) と [start2, end2) の共通部分があるかどうかを判定する。 """
  return start1 < end2 and start2 < end1

class DataModule10(L.LightningDataModule):
  def __init__(
      self,
      frames: int,
      frames_ref: int,
      n_refs: int,
      ref_max_kth: int,
      batch_size: int,
      n_batches: int,
      n_batches_val: int,
      same_density: bool,
      req: FeatsList | None = None,
      cache=True,
      num_workers=8,
      prefetch_factor=2,
  ):
    super().__init__()
    self.frames = frames
    self.frames_ref = frames_ref
    self.n_refs = n_refs
    self.ref_max_kth = ref_max_kth
    self.batch_size = batch_size
    self.n_batches = n_batches
    self.n_batches_val = n_batches_val
    self.same_density = same_density
    self.req = req
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor
    self.accessor = NPAccessor(cache=cache)

    self.intra_train: Any = None
    self.intra_valid: Any = None

  @property
  def _dataset_class(self):  # 継承したクラスで上書きしたいので、プロパティにしてみた
    return Dataset10

  def setup(self, stage: str | None = None):
    P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in P.dataset.speaker_ids]
    train_look = [LUT_ROOT / "parallel100" / sid for sid in P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in P.dataset.speaker_ids]
    speaker_ids = [i for i, _ in enumerate(P.dataset.speaker_ids)]

    train_look = [faiss.read_index(str(p / "soft.index")) for p in train_look]

    self._setup_datasets(train_dirs, valid_dirs, train_look, speaker_ids)

  def _setup_datasets(self, train_dirs: list[str], valid_dirs: list[str], train_look: list[Any], speaker_ids: list[int]):
    self.intra_train = self._dataset_class(
        accessor=self.accessor,
        dirs=train_dirs,
        dirs_ref=train_dirs,
        indices=train_look,
        speaker_ids=speaker_ids,
        frames=self.frames,
        frames_ref=self.frames_ref,
        n_refs=self.n_refs,
        ref_max_kth=self.ref_max_kth,
        rand_refs=28747632,
        rand_ref_kth=64378234,
        rand_start=35534253,
        shuffle=None,
        same_density=self.same_density,
        req=self.req,
    )
    self.intra_valid = self._dataset_class(
        accessor=self.accessor,
        dirs=valid_dirs,
        dirs_ref=train_dirs,
        indices=train_look,
        speaker_ids=speaker_ids,
        frames=self.frames,
        frames_ref=self.frames_ref,
        n_refs=self.n_refs,
        ref_max_kth=self.ref_max_kth,
        rand_refs=28747632,
        rand_ref_kth=64378234,
        rand_start=None,
        shuffle=7892639,
        same_density=self.same_density,
        req=self.req,
    )
    self.intra_valid = Subset(self.intra_valid, Random(37892834).choices(list(range(len(self.intra_valid))), k=self.n_batches_val * self.batch_size))

  def train_dataloader(self):
    return DataLoader(
        self.intra_train,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        sampler=RandomSampler(self.intra_train, replacement=True, num_samples=self.n_batches * self.batch_size),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=self.prefetch_factor,
    )

  def val_dataloader(self):
    return DataLoader(
        self.intra_valid,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=self.prefetch_factor,
    )

def bench():
  # dataloader がボトルネックになっていたので、調査のために。
  from tqdm import tqdm

  P.set_device("cuda")

  datamodule = DataModule10(
      frames=256,
      frames_ref=32,
      n_refs=32,
      ref_max_kth=32,
      batch_size=8,
      n_batches=200,
      n_batches_val=100,
      same_density=True,
      num_workers=0,
      prefetch_factor=None,
  )

  datamodule.setup()
  for _ in tqdm(datamodule.train_dataloader(), ncols=0, desc="Train"):
    pass

if __name__ == "__main__":
  bench()
