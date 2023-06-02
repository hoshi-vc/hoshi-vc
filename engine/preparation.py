# %%
import os
from functools import cached_property
from pathlib import Path

import faiss
import numpy as np
from autofaiss import build_index
from tqdm import tqdm

from engine.lib.dataset_jvs import JVS, JVSCategory
from engine.lib.extract import (Wav2Vec2, extract_melspec, extract_pitch_matrix, extract_pitch_topk)
from engine.lib.utils import (DATA_DIR, Device, NPArray, make_parents, np_safesave)
from engine.lib.vocoder import HiFiGAN

PITCH_TOPK = 8
CREPE_MODEL = "tiny"

FEATS_DIR = DATA_DIR / "attempt01" / "feats"
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

  @cached_property
  def extract_melspec(self):
    return extract_melspec

  @cached_property
  def extract_wav2vec2(self):
    return Wav2Vec2.load("facebook/wav2vec2-base").to(self.device)

  @cached_property
  def extract_pitch_matrix(self):
    return extract_pitch_matrix

  @cached_property
  def extract_pitch_topk(self):
    return extract_pitch_topk

  def prepare_feats(self):
    for category_id in ["parallel100", "nonpara30"]:
      for speaker_id in self.dataset.speaker_ids:
        DIR = FEATS_DIR / category_id / speaker_id
        W2V2 = DIR / "w2v2.npy"
        MEL = DIR / "mel.npy"
        PITCH_I = DIR / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy"
        PITCH_V = DIR / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy"

        if W2V2.exists() and MEL.exists() and PITCH_I.exists() and PITCH_V.exists(): return

        w2v2_list: list[NPArray] = []
        mel_list: list[NPArray] = []
        pitch_i_list: list[NPArray] = []
        pitch_v_list: list[NPArray] = []

        target_i = [i for i, x in enumerate(self.dataset_noaudio) if x.speaker_id == speaker_id and x.category_id == category_id]

        for i in tqdm(target_i, desc=f"Processing {speaker_id}", ncols=0):
          item = self.dataset[i]

          # Extract features
          audio, sr = item.audio[0], item.sr
          w2v2 = self.extract_wav2vec2(audio, sr)
          mel = self.extract_melspec(audio, sr)
          pitch_i, pitch_v = self.extract_pitch_topk(audio, sr, CREPE_MODEL, PITCH_TOPK, self.device)

          w2v2 = w2v2.cpu().numpy()
          mel = mel.cpu().numpy()
          pitch_i = pitch_i.cpu().numpy()
          pitch_v = pitch_v.cpu().numpy()

          # TODO: 本当は extract 関数の出力がすべて同じ長さになっていてほしい...
          if mel.shape[0] != w2v2.shape[0]:
            print(f"mel.shape[0] != w2v2.shape[0] :: {mel.shape[0]} != {w2v2.shape[0]} ({item.name})")
            w2v2 = pad_clip(mel, w2v2)
          if mel.shape[0] != pitch_i.shape[0]:
            if abs(mel.shape[0] - pitch_i.shape[0]) > 2:
              print(f"mel.shape[0] != pitch.shape[0] :: {mel.shape[0]} != {pitch_i.shape[0]} ({item.name})")
            pitch_i = pad_clip(mel, pitch_i)
            pitch_v = pad_clip(mel, pitch_v)

          assert w2v2.shape[0] == mel.shape[0] == pitch_i.shape[0] == pitch_v.shape[0]

          # Append to storage
          w2v2_list.append(w2v2)
          mel_list.append(mel)
          pitch_i_list.append(pitch_i)
          pitch_v_list.append(pitch_v)

        # print(w2v2_list[0].dtype, mel_list[0].dtype, pitch_i_list[0].dtype, pitch_v_list[0].dtype)

        DIR.mkdir(parents=True, exist_ok=True)
        np_safesave(W2V2, np.concatenate(w2v2_list))
        np_safesave(MEL, np.concatenate(mel_list))
        np_safesave(PITCH_I, np.concatenate(pitch_i_list))
        np_safesave(PITCH_V, np.concatenate(pitch_v_list))

        del w2v2_list
        del mel_list

  def prepare_faiss(self):
    for speaker_id in tqdm(self.dataset.speaker_ids, ncols=0, desc="Building index"):
      DEST = str(FAISS_DIR / f"{speaker_id}.index")
      if Path(DEST).exists(): continue

      indices = self.get_w2v2(speaker_id)

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

  def get_w2v2(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"w2v2.npy")

  def get_mel(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> NPArray:
    return np.load(FEATS_DIR / category_id / speaker_id / f"mel.npy")

  def get_pitch_topk(self, speaker_id: str, category_id: JVSCategory = "parallel100") -> tuple[NPArray, NPArray]:
    i = np.load(FEATS_DIR / category_id / speaker_id / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy")
    v = np.load(FEATS_DIR / category_id / speaker_id / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy")
    return i, v

  def get_index(self, speaker_id: str) -> faiss.IndexHNSWFlat:
    return faiss.read_index(str(FAISS_DIR / f"{speaker_id}.index"))

  @cached_property
  def vocoder(self):
    return HiFiGAN.load(DATA_DIR / "vocoder", download=True).to(self.device)

def pad_clip(reference: NPArray, value: NPArray) -> NPArray:
  assert reference.ndim == value.ndim == 2
  if reference.shape[0] > value.shape[0]:
    value = np.pad(value, ((0, reference.shape[0] - value.shape[0]), (0, 0)), mode="edge")
  elif reference.shape[0] < value.shape[0]:
    value = value[:reference.shape[0]]
  return value

if __name__ == "__main__":
  P = Preparation("cuda")
  P.prepare_feats()
  P.prepare_faiss()
