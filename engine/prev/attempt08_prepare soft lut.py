# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import os

import autofaiss
import faiss
import numpy as np
from tqdm import tqdm

from engine.singleton import Preparation
from engine.utils import DATA_DIR

LUT_ROOT = DATA_DIR / "attempt08" / "lookup"

def prepare_audio():
  for category_id in ["parallel100", "nonpara30"]:
    for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, desc=f"{category_id}", leave=False):
      SP_DIR = LUT_ROOT / category_id / speaker_id
      SOFT_FILE = SP_DIR / "soft.index"

      if SOFT_FILE.exists(): continue

      soft = P.get_hubert_soft(speaker_id, category_id)

      # PCA64,PQ64x4fs
      index: faiss.IndexPreTransform = faiss.index_factory(soft.shape[1], "PCA48,IVF128,Flat", faiss.METRIC_INNER_PRODUCT)
      faiss.downcast_index(index.index).nprobe = 4  # https://github.com/facebookresearch/faiss/issues/376
      index.train(soft.astype(np.float32))
      index.add(soft.astype(np.float32))
      assert index.is_trained

      # autofaiss.score_index(index, soft, save_on_disk=False)
      # raise Exception("stop")

      SP_DIR.mkdir(parents=True, exist_ok=True)
      faiss.write_index(index, str(SOFT_FILE) + ".tmp")
      os.replace(str(SOFT_FILE) + ".tmp", SOFT_FILE)

if __name__ == "__main__":
  P = Preparation("cuda")
  prepare_audio()
