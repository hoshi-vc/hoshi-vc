# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import os
from functools import cached_property
from pathlib import Path

import faiss
import librosa
import numpy as np
import torch
from autofaiss import build_index
from torch import Tensor
from tqdm import tqdm

from engine.lib.dataset_jvs import JVS, JVSCategory
from engine.lib.feats import Audio, Energy, MelSpec, Phoneme, Pitch, Wav2Vec2
from engine.lib.trim import trim_silence
from engine.lib.utils import (DATA_DIR, Device, NPArray, make_parents, np_safesave)
from engine.lib.vocoder import HiFiGAN

PITCH_TOPK = 8
CREPE_MODEL = "tiny"

PHONEME_TOPK = 8

FEATS_DIR = DATA_DIR / "feats"
FAISS_DIR = DATA_DIR / "attempt01" / "faiss"

class Preparation:
  def __init__(self, device: Device):
    self.device = device

  @cached_property
  def dataset(self):
    return JVS(DATA_DIR / "datasets", download=True)

  @cached_property
  def dataset_noaudio(self):
    return JVS(DATA_DIR / "datasets", download=True, no_audio=True)

  def normalize_audio(self, audio: Tensor, sr: int, trim=True, normalize=True):
    device = self.device
    audio = audio.numpy()
    if trim: audio = trim_silence(audio, sr)
    if normalize: audio = librosa.util.normalize(audio) * 0.95
    audio = torch.as_tensor(audio, device=device)
    return audio

  @cached_property
  def extract_audio(self):
    return Audio()

  @cached_property
  def extract_melspec(self):
    return MelSpec()

  @cached_property
  def extract_energy(self):
    return Energy()

  @cached_property
  def extract_pitch(self):
    return Pitch(CREPE_MODEL, PITCH_TOPK)

  @cached_property
  def extract_wav2vec2(self):
    return Wav2Vec2.load("facebook/wav2vec2-base").to(self.device)

  @cached_property
  def extract_phoneme(self):
    return Phoneme.load("facebook/wav2vec2-xlsr-53-espeak-cv-ft", PHONEME_TOPK).to(self.device)

  def prepare_feats(self):
    for category_id in ["parallel100", "nonpara30"]:
      for speaker_id in self.dataset.speaker_ids:
        DIR = FEATS_DIR / category_id / speaker_id
        AUDIO = DIR / "audio.npy"
        MELSPEC = DIR / "melspec.npy"
        ENERGY = DIR / "energy.npy"
        PITCH_I = DIR / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy"
        PITCH_V = DIR / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy"
        WAV2VEC2 = DIR / "wav2vec2.npy"
        PHONEME_I = DIR / f"phoneme_i_{PHONEME_TOPK}.npy"
        PHONEME_V = DIR / f"phoneme_v_{PHONEME_TOPK}.npy"

        if AUDIO.exists() and MELSPEC.exists() and ENERGY.exists() and PITCH_I.exists() and PITCH_V.exists() and WAV2VEC2.exists() and PHONEME_I.exists(
        ) and PHONEME_V.exists():
          continue

        audio_list: list[NPArray] = []
        mel_list: list[NPArray] = []
        energy_list: list[NPArray] = []
        w2v2_list: list[NPArray] = []
        phoneme_i_list: list[NPArray] = []
        phoneme_v_list: list[NPArray] = []
        pitch_i_list: list[NPArray] = []
        pitch_v_list: list[NPArray] = []

        target_i = [i for i, x in enumerate(self.dataset_noaudio) if x.speaker_id == speaker_id and x.category_id == category_id]

        for i in tqdm(target_i, desc=f"Processing {speaker_id}", ncols=0):
          item = self.dataset[i]

          # normalize
          audio, sr = item.audio[0], item.sr
          audio = self.normalize_audio(audio, sr)

          # extract features
          audio_feat = self.extract_audio(audio, sr)
          mel = self.extract_melspec(audio, sr)
          energy = self.extract_energy(audio, sr)
          wav2vec2 = self.extract_wav2vec2(audio, sr)
          phoneme_i, phoneme_v = self.extract_phoneme(audio, sr)
          pitch_i, pitch_v = self.extract_pitch(audio, sr)

          audio_feat = audio_feat.cpu().numpy()
          mel = mel.cpu().numpy()
          energy = energy.cpu().numpy()
          wav2vec2 = wav2vec2.cpu().numpy()
          phoneme_i = phoneme_i.cpu().numpy()
          phoneme_v = phoneme_v.cpu().numpy()
          pitch_i = pitch_i.cpu().numpy()
          pitch_v = pitch_v.cpu().numpy()

          # append to storage
          audio_list.append(audio_feat)
          mel_list.append(mel)
          energy_list.append(energy)
          w2v2_list.append(wav2vec2)
          phoneme_i_list.append(phoneme_i)
          phoneme_v_list.append(phoneme_v)
          pitch_i_list.append(pitch_i)
          pitch_v_list.append(pitch_v)

        # print(w2v2_list[0].dtype, mel_list[0].dtype, pitch_i_list[0].dtype, pitch_v_list[0].dtype)

        DIR.mkdir(parents=True, exist_ok=True)
        np_safesave(AUDIO, np.concatenate(audio_list))
        np_safesave(MELSPEC, np.concatenate(mel_list))
        np_safesave(ENERGY, np.concatenate(energy_list))
        np_safesave(WAV2VEC2, np.concatenate(w2v2_list))
        np_safesave(PHONEME_I, np.concatenate(phoneme_i_list))
        np_safesave(PHONEME_V, np.concatenate(phoneme_v_list))
        np_safesave(PITCH_I, np.concatenate(pitch_i_list))
        np_safesave(PITCH_V, np.concatenate(pitch_v_list))

  def prepare_faiss(self):
    for speaker_id in tqdm(self.dataset.speaker_ids, ncols=0, desc="Building index"):
      DEST = str(FAISS_DIR / f"{speaker_id}.index")
      if Path(DEST).exists(): continue

      indices = self.get_wav2vec2(speaker_id)

      # https://criteo.github.io/autofaiss/_source/autofaiss.external.html#autofaiss.external.quantize.build_index
      # index, index_infos = build_index(
      #     indices,
      #     save_on_disk=True,
      #     index_path=str(FAISS_DIR / f"{speaker_id}_knn.index"),
      #     index_infos_path=str(FAISS_DIR / f"{speaker_id}_infos.json"),
      #     metric_type="ip",
      #     max_index_query_time_ms=1,
      #     max_index_memory_usage="200MB",
      #     min_nearest_neighbors_to_retrieve=16,
      # )

      index: faiss.IndexHNSWFlat = faiss.index_factory(indices.shape[1], "HNSW32", faiss.METRIC_INNER_PRODUCT)
      index.hnsw.efSearch = 300
      index.add(indices)
      assert index.is_trained

      make_parents(DEST)
      faiss.write_index(index, DEST + ".tmp")
      os.replace(DEST + ".tmp", DEST)

      del indices

  def get_audio(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"audio.npy")

  def get_melspec(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"melspec.npy")

  def get_energy(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"energy.npy")

  def get_pitch(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> tuple[NPArray, NPArray]:
    i = np.load(FEATS_DIR / category_id / speaker_id / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy")
    v = np.load(FEATS_DIR / category_id / speaker_id / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy")
    return i, v

  def get_wav2vec2(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"wav2vec2.npy")

  def get_phoneme(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> tuple[NPArray, NPArray]:
    i = np.load(FEATS_DIR / category_id / speaker_id / f"phoneme_i_{PHONEME_TOPK}.npy")
    v = np.load(FEATS_DIR / category_id / speaker_id / f"phoneme_v_{PHONEME_TOPK}.npy")
    return i, v

  def get_index(self, speaker_id: str) -> faiss.IndexHNSWFlat:
    return faiss.read_index(str(FAISS_DIR / f"{speaker_id}.index"))

  @cached_property
  def vocoder(self):
    return HiFiGAN.load(DATA_DIR / "vocoder", download=True).to(self.device)

if __name__ == "__main__":
  P = Preparation("cuda")
  P.prepare_feats()
  P.prepare_faiss()

def check_feats():
  pass
  # %%
  # ちゃんと特徴たちがアラインしているかを目視で確認したくて書いた

  from matplotlib import pyplot as plt

  s = 33900
  s = 0
  e = s + 200
  mel = P.get_melspec("jvs001")[s:e]
  energy = P.get_energy("jvs001")[s:e]
  w2v2 = P.get_wav2vec2("jvs001")[s:e]
  phoneme_i, phoneme_v = P.get_phoneme("jvs001")
  pitch_i, pitch_v = P.get_pitch("jvs001")
  phoneme_i = phoneme_i[s:e]
  phoneme_v = phoneme_v[s:e]
  pitch_i = pitch_i[s:e]
  pitch_v = pitch_v[s:e]

  plt.pcolormesh(mel.T), plt.show()
  plt.plot(energy), plt.xlim(0, e - s), plt.show()
  plt.pcolormesh(np.flip(np.dot(w2v2, w2v2.T), 0)), plt.show()
  plt.plot(pitch_i[:, 0]), plt.xlim(0, e - s), plt.show()
  plt.plot(pitch_v[:, 0]), plt.xlim(0, e - s), plt.show()
  plt.plot(phoneme_i[:, 0] == 0), plt.xlim(0, e - s), plt.show()
  plt.plot(phoneme_v[:, 0]), plt.xlim(0, e - s), plt.show()
