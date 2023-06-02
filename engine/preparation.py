from functools import cached_property

import faiss
import numpy as np
from autofaiss import build_index
from tqdm import tqdm

from engine.lib.dataset_jvs import JVS
from engine.lib.extract import Wav2Vec2, extract_melspec
from engine.lib.utils import DATA_DIR, Device, NPArray
from engine.lib.vocoder import HiFiGAN

KV_DIR = DATA_DIR / "attempt01" / "kv_pairs"
FAISS_DIR = DATA_DIR / "attempt01" / "faiss"

class Preparation:
  def __init__(self, device: Device):
    self.device = device

  @cached_property
  def dataset(self):
    return JVS(DATA_DIR / "datasets", download=True)

  @cached_property
  def extract_melspec(self):
    return extract_melspec

  @cached_property
  def extract_wav2vec2(self):
    return Wav2Vec2.load("facebook/wav2vec2-base").to(self.device)

  def prepare_kv_pairs(self):
    if not (KV_DIR / ".done").exists():
      KV_DIR.mkdir(parents=True, exist_ok=True)

      speaker_ids = [item.speaker_id for item in tqdm(self.dataset, ncols=0)]

      for speaker_id in self.dataset.speaker_ids:
        indices: list[NPArray] = []
        values: list[NPArray] = []

        target_i = [i for i, x in enumerate(speaker_ids) if x == speaker_id]

        with tqdm(target_i, desc=f"Processing {speaker_id}", ncols=0) as bar:
          for i in target_i:
            item = self.dataset[i]
            # Skip & update bar
            if item.category_id != "parallel100":
              bar.total -= 1
              bar.refresh()
              continue

            # Extract features
            audio, sr = item.audio[0], item.sr
            wav2vec2 = self.extract_wav2vec2(audio, sr)
            mel = self.extract_melspec(audio, sr)

            # Append to storage
            indices.append(wav2vec2.cpu().numpy())
            values.append(mel.cpu().numpy())

            assert wav2vec2.shape[0] == mel.shape[0]

            # Update bar
            bar.update(1)

          KV_DIR.mkdir(parents=True, exist_ok=True)
          np.save(KV_DIR / f"{speaker_id}_index.npy", np.concatenate(indices))
          np.save(KV_DIR / f"{speaker_id}_value.npy", np.concatenate(values))

        del indices
        del values

      (KV_DIR / ".done").touch()

  def prepare_faiss(self):
    if not (FAISS_DIR / ".done").exists():
      FAISS_DIR.mkdir(parents=True, exist_ok=True)

      for speaker_id in tqdm(self.dataset.speaker_ids, ncols=0, desc="Building index"):
        indices = np.load(KV_DIR / f"{speaker_id}_index.npy")

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
        faiss.write_index(index, str(FAISS_DIR / f"{speaker_id}.index"))

        del indices

      (FAISS_DIR / ".done").touch()

  def get_keys(self, speaker_id: str):
    return np.load(KV_DIR / f"{speaker_id}_index.npy")

  def get_values(self, speaker_id: str):
    return np.load(KV_DIR / f"{speaker_id}_value.npy")

  def get_index(self, speaker_id: str):
    return faiss.read_index(str(FAISS_DIR / f"{speaker_id}.index"))

  @cached_property
  def vocoder(self):
    return HiFiGAN.load(DATA_DIR / "vocoder", download=True).to(self.device)
