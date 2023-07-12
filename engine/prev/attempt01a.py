# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import os
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from engine.lib.utils import NPArray
from engine.lib.utils_ui import play_audio, plot_spectrogram
from engine.singleton import Preparation
from engine.utils import DATA_DIR

LUT_ROOT = DATA_DIR / "attempt01a" / "lookup"

def save_index(features: NPArray, dest: Path):
  index: faiss.IndexPreTransform = faiss.index_factory(features.shape[1], "Flat", faiss.METRIC_L1)
  index.train(np.ascontiguousarray(features.astype(np.float32)))
  index.add(np.ascontiguousarray(features.astype(np.float32)))
  assert index.is_trained

  dest.parent.mkdir(parents=True, exist_ok=True)
  faiss.write_index(index, str(dest) + ".tmp")
  os.replace(str(dest) + ".tmp", dest)

def prepare():
  for category_id in ["parallel100"]:
    for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, desc=f"{category_id}", leave=False):
      SP_DIR = LUT_ROOT / category_id / speaker_id
      MEL_FILE = SP_DIR / "melspec.index"
      SOFT_FILE = SP_DIR / "hubert_soft.index"

      if MEL_FILE.exists(): continue

      mel = P.get_melspec(speaker_id, category_id)
      soft = P.get_hubert_soft(speaker_id, category_id)

      save_index(mel, MEL_FILE)
      save_index(soft, SOFT_FILE)

if __name__ == "__main__":
  P = Preparation("cuda")
  P.prepare_feats()
  prepare()

  # %%

  target_speaker = "jvs001"
  index = faiss.read_index(str(LUT_ROOT / "parallel100" / target_speaker / "hubert_soft.index"))
  target_mel = P.get_melspec(target_speaker)
  target_pitch_i = P.get_pitch(target_speaker)[0]

  item = P.dataset[1000]
  print(item.name)

  audio, sr = item.audio[0], item.sr
  mel = P.extract_melspec(audio, sr)
  pitch_i = P.extract_pitch(audio, sr)[0] - 30
  keys = P.extract_hubert_soft(audio, sr).cpu().numpy()
  keys = np.ascontiguousarray(keys).astype(np.float32)

  top_k = 32

  if False:
    hat = []
    for i in range(len(keys)):
      D, I = index.search(keys[None, i], top_k)
      items: list[NPArray] = []
      for j in range(top_k):
        items.append(target_mel[I[0][j]])
      hat.append(np.mean(np.stack(items), axis=0))
    hat = torch.as_tensor(np.vstack(hat))
  else:
    frames = []
    for i in range(len(keys)):
      D, I = index.search(keys[None, i], top_k)
      items: list[tuple[int, NPArray]] = []
      src_pitch = pitch_i[i][0]
      for j in range(top_k):
        ref_pitch = target_pitch_i[I[0][j]][0]
        items.append((abs(src_pitch - ref_pitch), target_mel[I[0][j]], ref_pitch))
      items.sort(key=lambda x: x[0])
      frames.append(items[0])
    hat = torch.as_tensor(np.vstack([x[1] for x in frames]))
    pitch = np.vstack([x[2] for x in frames])

  audio_hat, sr_hat = P.vocoder(hat)
  plot_spectrogram(mel.T)
  plot_spectrogram(hat.T)
  plt.plot(pitch)

  play_audio(audio, sr)
  play_audio(audio_hat, sr_hat)
