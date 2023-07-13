# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import shutil
from posixpath import relpath, sep
from typing import TYPE_CHECKING

import faiss
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate
from tqdm import tqdm

from engine.attempts.a10_dataset import (DataModule10, Dataset10, Feats10, NPAccessor)
from engine.lib.utils import np_safesave
from engine.singleton import DATA_DIR, FEATS_DIR, P

if TYPE_CHECKING: from engine.attempts.a11a import VCModule

LUT_ROOT = DATA_DIR / "a11" / "lookup"

class Dataset11A(Dataset10):
  def _get_key(self, d: str, speaker_id: int, start: int, frames: int, feats: Feats10, key_indices: list[int]) -> Tensor:
    p = str(LUT_ROOT) + sep + relpath(d, FEATS_DIR) + sep + "keys.npy"
    end = start + frames
    keys = np.array(self.accessor.load(p)[start:end])
    return keys[key_indices]

class DataModule11A(DataModule10):
  @property
  def _dataset_class(self):
    return Dataset11A

  def setup(self, stage: str | None = None):
    P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in P.dataset.speaker_ids]
    train_look = [LUT_ROOT / "parallel100" / sid for sid in P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in P.dataset.speaker_ids]
    speaker_ids = [i for i, _ in enumerate(P.dataset.speaker_ids)]

    train_look = [faiss.read_index(str(p / "keys.index")) for p in train_look]

    self._setup_datasets(train_dirs, valid_dirs, train_look, speaker_ids)

def prepare_key_index(module: "VCModule", resume: bool = False):
  if not resume: shutil.rmtree(LUT_ROOT, ignore_errors=True)
  LUT_ROOT.mkdir(parents=True, exist_ok=True)

  accessor = NPAccessor(cache=True)

  for category_id in ["parallel100", "nonpara30"]:
    for speaker_id in P.dataset.speaker_ids:
      SP_DIR = LUT_ROOT / category_id / speaker_id
      KEY_FILE = SP_DIR / "keys.npy"
      KEY_INDEX = SP_DIR / "keys.index"

      if KEY_FILE.exists() and KEY_INDEX.exists(): continue

      frame_len = 256

      feat_dir = FEATS_DIR / category_id / speaker_id
      speaker_id_num = P.dataset.speaker_ids.index(speaker_id)
      feat_length = np.load(feat_dir / "melspec.npy", mmap_mode="r").shape[0]
      total_batches = feat_length // frame_len

      with torch.inference_mode():
        _device = module.device
        module.to(P.device)
        keys = []
        for batch_idx in tqdm(range(total_batches), ncols=0, desc=f"{category_id}/{speaker_id}", leave=False):
          start = batch_idx * frame_len
          actual_len = frame_len if batch_idx < total_batches - 1 else feat_length - start

          batch = Feats10.load(accessor, feat_dir, speaker_id_num, start, actual_len)
          batch = default_collate([batch])
          batch: Feats10 = module.transfer_batch_to_device(batch, P.device, 0)

          src_energy = module.vc_model.forward_energy(batch.energy.float())
          src_pitch = module.vc_model.forward_pitch(batch.pitch_i)
          src_soft = module.vc_model.forward_w2v2(batch.soft.float())
          src_key = module.vc_model.forward_key(src_energy, src_pitch, src_soft)

          keys.append(src_key.squeeze().cpu().numpy())
        keys = np.concatenate(keys, axis=0)
        module.to(_device)

      assert len(keys) == len(P.get_melspec(speaker_id, category_id))

      # PCA64,PQ64x4fs
      index: faiss.IndexPreTransform = faiss.index_factory(keys.shape[1], "PCA48,IVF128,Flat", faiss.METRIC_INNER_PRODUCT)
      faiss.downcast_index(index.index).nprobe = 4  # https://github.com/facebookresearch/faiss/issues/376
      index.train(keys.astype(np.float32))
      index.add(keys.astype(np.float32))
      assert index.is_trained

      SP_DIR.mkdir(parents=True, exist_ok=True)
      np_safesave(KEY_FILE, keys)
      faiss.write_index(index, str(KEY_INDEX) + ".tmp")
      os.replace(str(KEY_INDEX) + ".tmp", KEY_INDEX)

if __name__ == "__main__":
  from engine.attempts.a10 import VCModule
  P.set_device("cuda")

  CKPT = DATA_DIR / "a11/checkpoints/worthy-cloud-5/686bjwic/last.ckpt"
  prepare_key_index(VCModule.load_from_checkpoint(CKPT, map_location=P.device))
