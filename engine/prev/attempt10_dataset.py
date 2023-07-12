# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path
from random import Random
from typing import Any, NamedTuple, Optional

import faiss
import lightning.pytorch as L
import numpy as np
from faiss import IndexPreTransform
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from engine.attempt08_prepare import LUT_ROOT
from engine.lib.utils import NPArray, clamp
from engine.singleton import (CREPE_MODEL, FEATS_DIR, PHONEME_TOPK, PITCH_TOPK, Preparation)

class Feats09(NamedTuple):
  audio: Tensor
  speaker: Tensor
  energy: Tensor
  mel: Tensor
  phoneme_i: Tensor
  phoneme_v: Tensor
  pitch_i: Tensor
  pitch_v: Tensor
  soft: Tensor

class Entry09(NamedTuple):
  src: Feats09
  ref: list[Feats09]
  tgt_speaker: Tensor
  tgt_ref: list[Feats09]

class NPAccessor:
  def __init__(self, cache: bool):
    self.cache = cache
    self.cache_dict: dict[str, NPArray] = {}

  def load(self, p: Path) -> NPArray:
    p = p.resolve()
    if p in self.cache_dict: return self.cache_dict[p]
    if self.cache:
      self.cache_dict[p] = np.load(p, mmap_mode="r")

      # 各プロセスごとに独立した cache_dict が作られてた
      # print(getpid(), len(self.cache_dict))

      return self.cache_dict[p]
    return np.load(p, mmap_mode="r")

  def clear_cache(self) -> None:
    self.cache_dict.clear()

class Dataset09(Dataset):
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
      rand_tgt: int,
      same_density=False,
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

    self.starts: list[tuple[str, str, IndexPreTransform, int, int]] = []
    self.max_ref_lens: dict[str, int] = {}
    for d, dref, faiss, speaker_id in zip(dirs, dirs_ref, indices, speaker_ids):
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      self.max_ref_lens[dref] = np.load(dref / "melspec.npy", mmap_mode="r").shape[0]
      for start in range(0, length - frames * 2, frames):
        self.starts.append((d, dref, faiss, speaker_id, start))

    if shuffle: Random(shuffle).shuffle(self.starts)

    self.rand_tgt = Random(rand_tgt)

  def load_entry(self, d: str, speaker_id: int, start: int, frames: int) -> Feats09:
    end = start + frames

    return Feats09(
        audio=np.array(self._load(d / "audio.npy")[start * 256:end * 256]),
        speaker=np.array([speaker_id]),
        energy=np.array(self._load(d / "energy.npy")[start:end]),
        mel=np.array(self._load(d / "melspec.npy")[start:end]),
        phoneme_i=np.array(self._load(d / f"phoneme_i_{PHONEME_TOPK}.npy")[start:end], np.int64),
        phoneme_v=np.array(self._load(d / f"phoneme_v_{PHONEME_TOPK}.npy")[start:end]),
        pitch_i=np.array(self._load(d / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy")[start:end], np.int64),
        pitch_v=np.array(self._load(d / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy")[start:end]),
        soft=np.array(self._load(d / "hubert_soft.npy")[start:end]),
    )

  def _load(self, file: Path) -> Any:
    # return np.load(file, mmap_mode="r")
    return self.accessor.load(file)

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> Entry09:
    d, dref, faiss, speaker_id, start = self.starts[index]
    if self.rand_start: start += self.rand_start.randint(0, self.frames)

    max_ref_len = self.max_ref_lens[dref]

    src = self.load_entry(d, speaker_id, start, self.frames)

    if not self.same_density:
      key_indices = self.rand_refs.sample(range(len(src.soft)), self.n_refs)
      key_indices.sort()
    else:
      key_indices = np.linspace(0, len(src.soft) - self.frames_ref * 2, self.n_refs, dtype=np.int64)
      key_indices += self.rand_refs.randint(0, self.frames_ref)

    keys = src.soft[key_indices]

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

    refs: list[Feats09] = []
    for ref_start in ref_starts:
      ref = self.load_entry(dref, speaker_id, ref_start, self.frames_ref)
      refs.append(ref)

    # === tgt_refs

    tgt_speaker_id = speaker_id
    while tgt_speaker_id == speaker_id:
      tgt_d, tgt_dref, tgt_faiss, tgt_speaker_id, _ = self.rand_tgt.choice(self.starts)
    tgt_max_ref_len = self.max_ref_lens[tgt_dref]
    _, tgt_ref_indices = tgt_faiss.search(keys.astype(np.float32), self.ref_max_kth)

    ref_starts = []
    for i in range(self.n_refs):
      can_be: list[int] = []
      ref_start: int | None = None
      for k in range(self.ref_max_kth):
        candidate_middle = tgt_ref_indices[i, k]
        candidate_start = clamp(candidate_middle - self.frames_ref // 2, 0, tgt_max_ref_len - self.frames_ref)

        can_be.append(candidate_start)

        margin = self.frames_ref // 4
        if any(ref_start + margin <= candidate_middle < ref_start + self.frames_ref - margin for ref_start in ref_starts): continue

        ref_start = candidate_start
        break

      if ref_start is None:
        if len(can_be) > 0:
          ref_start = can_be[0]
        elif i > 0:
          ref_start = ref_start[-1]
        else:
          print(f"Error: No suitable reference for target speaker found. Please increase ref_max_kth. (spk={speaker_id}, start={start})")
          ref_start = 0

      ref_starts.append(ref_start)

    assert len(ref_starts) == self.n_refs

    tgt_refs: list[Feats09] = []
    for ref_start in ref_starts:
      ref = self.load_entry(tgt_dref, tgt_speaker_id, ref_start, self.frames_ref)
      tgt_refs.append(ref)

    # === Return

    return Entry09(src, refs, np.array([tgt_speaker_id]), tgt_refs)

def intersect(start1: int, end1: int, start2: int, end2: int) -> bool:
  """ [start1, end1) と [start2, end2) の共通部分があるかどうかを判定する。 """
  return start1 < end2 and start2 < end1

class DataModule09(L.LightningDataModule):
  def __init__(
      self,
      P: Preparation,
      frames: int,
      frames_ref: int,
      n_refs: int,
      ref_max_kth: int,
      batch_size: int,
      n_batches: int,
      n_batches_val: int,
      same_density=False,
      cache=True,
      num_workers=8,
      prefetch_factor=2,
  ):
    super().__init__()
    self.P = P
    self.frames = frames
    self.frames_ref = frames_ref
    self.n_refs = n_refs
    self.ref_max_kth = ref_max_kth
    self.batch_size = batch_size
    self.n_batches = n_batches
    self.n_batches_val = n_batches_val
    self.same_density = same_density
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor
    self.accessor = NPAccessor(cache=cache)

    self.intra_train: Any = None
    self.intra_valid: Any = None

  def setup(self, stage: str):
    self.P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    train_look = [LUT_ROOT / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    # valid_look = [LUT_ROOT / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    speaker_ids = [i for i, _ in enumerate(self.P.dataset.speaker_ids)]

    train_look = [faiss.read_index(str(p / "soft.index")) for p in train_look]
    # valid_look = [faiss.read_index(str(p / "soft.index")) for p in valid_look]

    self.intra_train = Dataset09(
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
        rand_tgt=29836458,
        same_density=self.same_density,
    )
    self.intra_valid = Dataset09(
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
        rand_tgt=29836458,
        same_density=self.same_density,
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
