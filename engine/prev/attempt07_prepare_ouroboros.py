# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# 仮定: 学習時の IntraDomainDataset において...
#      - start_hop == frames
#      - not random_offset
# これらの仮定は、学習データセットの作成時に変換チャンクの切れ端部分の精度が低くなりうること（ Conv1d の paddding ）
# そして、一つの変換元音声バッチに複数の話者が含まれうることが理由。
# 前者については変換ウィンドウを重なりをもたせながらずらして切れ目をフェードインさせれば良さそうだけど、面倒なので。

from random import Random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

import engine.prev.attempt10 as Attempt
from engine.lib.utils import NPArray, np_safesave
from engine.prev.attempt10_dataset import Entry09, Feats09
from engine.singleton import CREPE_MODEL, DATA_DIR, FEATS_DIR, PITCH_TOPK, P

CKPT = DATA_DIR / "attempt07/checkpoints/fine-lion-1/7p0mdwvn/last.ckpt"

AUG_ROOT = DATA_DIR / "attempt07a-stage1" / CKPT.relative_to(DATA_DIR).with_suffix("")

SPKSIM_THRESHOLD1 = 0.6  # 変換先話者に近い
SPKSIM_THRESHOLD2 = 0.2  # 変換元話者にもある程度近い （大きく異なる話者の変換では精度が下がるのでひとまず避けとく）
SPKSIM_RETRY = 20

class FeatureDataset(Dataset):
  def __init__(self, dirs: list[str], speaker_ids: list[int], frames: int):
    start_hop = frames

    assert len(dirs) == len(speaker_ids)
    self.frames = frames

    self.starts: list[tuple[str, int, int]] = []
    self.same_domain_lut: dict[str, list[int]] = {}
    for d, speaker_id in zip(dirs, speaker_ids):
      length = np.load(d / "melspec.npy", mmap_mode="r").shape[0]
      for start in range(0, length - frames - start_hop, start_hop):
        self.starts.append((d, speaker_id, start))
        self.same_domain_lut.setdefault(d, []).append(len(self.starts) - 1)

  def __len__(self) -> int:
    return len(self.starts)

  def __getitem__(self, index: int) -> Entry09:
    d, speaker_id, start = self.starts[index]

    # TODO: 面倒なので直接呼んでる
    return IntraDomainDataset4.load_entry(None, d, speaker_id, start, self.frames)

def load_ref_entry(speaker: str, frames=None) -> Feats09:
  frames = frames or START_HOP * 8

  feat_dir = FEATS_DIR / "parallel100" / speaker
  speaker_id = P.dataset.speaker_ids.index(speaker)

  # TODO: 面倒なので直接呼んでる
  entry = IntraDomainDataset4.load_entry(None, feat_dir, speaker_id, 0, frames)

  entry: Feats09 = default_collate([entry])
  entry: Feats09 = model.transfer_batch_to_device(entry, model.device, 0)

  return entry

def calc_mean_pitch(entry: Feats09) -> NPArray:
  mask = entry.pitch_v > 0.5
  return (entry.pitch_i * mask).sum() / mask.sum()

class Shuffler:
  def __init__(self, items: list, rand: Random):
    self.items = items
    self.rand = rand
    self.current = self._shuffled()
    self.index = 0

  def _shuffled(self):
    items = self.items[:]
    self.rand.shuffle(items)
    return items

  def next(self):
    if self.index >= len(self.current):
      self.current = self._shuffled()
      self.index = 0

    item = self.current[self.index]
    self.index += 1
    return item

def prepare_audio():
  ref_map: dict[str, Feats09] = {}
  pitch_map: dict[str, NPArray] = {}
  spkemb_map: dict[str, Tensor] = {}
  ref_rand = Random(67648768)

  for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, leave=False, desc="Preparing"):
    ref_map[speaker_id] = load_ref_entry(speaker_id)
    pitch_map[speaker_id] = calc_mean_pitch(ref_map[speaker_id])
    spkemb_map[speaker_id] = P.spkemb(ref_map[speaker_id].audio.float(), 22050)

  for category_id in ["parallel100", "nonpara30"]:
    for speaker_id in P.dataset.speaker_ids:
      SP_DIR = AUG_ROOT / category_id / speaker_id
      AUDIO_FILE = SP_DIR / "audio.npy"

      if AUDIO_FILE.exists(): continue

      # 同一バッチは同じリファレンス話者になるので、多様性のために batch_size=1 にしておく
      feat_dirs = [FEATS_DIR / category_id / speaker_id]
      speaker_ids = [P.dataset.speaker_ids.index(speaker_id)]
      dataset = FeatureDataset(feat_dirs, speaker_ids, START_HOP)
      loader = DataLoader(dataset, batch_size=1, num_workers=4)

      # 話している内容とリファレンス話者の結びつきは避けたいので、ターゲット話者ごとにシャッフルし直す
      ref_speakers = Shuffler(P.dataset.speaker_ids, ref_rand)

      audios = []
      for i, batch in tqdm(enumerate(loader), total=len(loader), ncols=0, desc=f"Loading {speaker_id}", leave=False):
        with torch.inference_mode():
          batch: Feats09 = model.transfer_batch_to_device(batch, model.device, 0)

          audio = None
          audio_score_map: list[tuple[float, Tensor, Any]] = []
          for k in range(SPKSIM_RETRY):
            ref_speaker = ref_speakers.next()
            ref_entry = ref_map[ref_speaker]
            src_pitch_shift = pitch_map[ref_speaker] - pitch_map[speaker_id]

            src_energy = model.vc_model.forward_energy(batch.energy.float())
            src_pitch = model.vc_model.forward_pitch((batch.pitch_i + src_pitch_shift).clamp(0, 359).long())
            src_w2v2 = model.vc_model.forward_w2v2(batch.soft.float())
            src_mel = model.vc_model.forward_mel(batch.mel.float())
            src_key = model.vc_model.forward_key(src_energy, src_pitch, src_w2v2)

            ref_energy = model.vc_model.forward_energy(ref_entry.energy.float())
            ref_pitch = model.vc_model.forward_pitch(ref_entry.pitch_i.long())
            ref_w2v2 = model.vc_model.forward_w2v2(ref_entry.soft.float())
            ref_mel = model.vc_model.forward_mel(ref_entry.mel.float())
            ref_key = model.vc_model.forward_key(ref_energy, ref_pitch, ref_w2v2).repeat(len(src_key), 1, 1)
            ref_value = model.vc_model.forward_value(ref_energy, ref_pitch, ref_mel).repeat(len(src_key), 1, 1)

            tgt_value, _ = model.vc_model.lookup(src_key, ref_key, ref_value)

            tgt_mel = model.vc_model.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

            tgt_audio = model.vocoder_forward(tgt_mel).reshape(-1).to(torch.float16)

            tgt_audio_spkemb = P.spkemb(tgt_audio.float(), 22050)

            spksim1 = torch.cosine_similarity(spkemb_map[ref_speaker], tgt_audio_spkemb).item()
            spksim2 = torch.cosine_similarity(spkemb_map[speaker_id], tgt_audio_spkemb).item()

            spksim = min(1, spksim1 / SPKSIM_THRESHOLD1) + min(1, spksim2 / SPKSIM_THRESHOLD2)  # TODO

            if spksim1 > SPKSIM_THRESHOLD1 and spksim2 > SPKSIM_THRESHOLD2:
              audio = tgt_audio
              # if k > 0: tqdm.write(f"Generated audio for {speaker_id} at {i}th batch (spksim={spksim:.3f}, retry={k})")
              break

            audio_score_map.append((spksim, tgt_audio, (spksim1, spksim2)))

          if audio is None:
            audio_score_map.sort(key=lambda x: x[0], reverse=True)
            audio = audio_score_map[0][1]
            tqdm.write(
                f"Warning: Failed to generate audio with enough spksim for {speaker_id} at {i}th batch (spksim={audio_score_map[0][2][0]:.3f}, {audio_score_map[0][2][1]:.3f})"
            )

          audios.append(audio.cpu().numpy())

      audios = np.concatenate(audios, axis=0)

      SP_DIR.mkdir(parents=True, exist_ok=True)

      np_safesave(AUDIO_FILE, audios)

def prepare_feats():
  # prepare.py からコピーして、書き換えた
  # TODO: リファクタリングする

  for category_id in ["parallel100", "nonpara30"]:
    for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, leave=False):
      DIR = AUG_ROOT / category_id / speaker_id
      AUDIO = DIR / "audio.npy"
      MELSPEC = DIR / "melspec.npy"
      ENERGY = DIR / "energy.npy"
      PITCH_I = DIR / f"pitch_i_{CREPE_MODEL}_{PITCH_TOPK}.npy"
      PITCH_V = DIR / f"pitch_v_{CREPE_MODEL}_{PITCH_TOPK}.npy"
      HUBERT_SOFT = DIR / "hubert_soft.npy"

      if AUDIO.exists() and MELSPEC.exists() and ENERGY.exists() and PITCH_I.exists() and PITCH_V.exists() and HUBERT_SOFT.exists():
        continue

      audio_long = np.load(DIR / "audio.npy", mmap_mode="r")

      mel_list: list[NPArray] = []
      energy_list: list[NPArray] = []
      pitch_i_list: list[NPArray] = []
      pitch_v_list: list[NPArray] = []
      hubert_soft_list: list[NPArray] = []

      for start in tqdm(range(0, len(audio_long) - AUDIO_HOP, AUDIO_HOP), ncols=0, desc=f"Processing {speaker_id}", leave=False):
        with torch.inference_mode():
          audio, sr = audio_long[start:start + AUDIO_HOP], 22050
          audio = torch.as_tensor(audio.copy(), dtype=torch.float32, device=P.device)

          mel = P.extract_melspec(audio, sr)
          energy = P.extract_energy(audio, sr)
          pitch_i, pitch_v = P.extract_pitch(audio, sr)
          hubert_soft = P.extract_hubert_soft(audio, sr)

          mel = mel.cpu().numpy()
          energy = energy.cpu().numpy()
          pitch_i = pitch_i.cpu().numpy()
          pitch_v = pitch_v.cpu().numpy()
          hubert_soft = hubert_soft.cpu().numpy()

          mel_list.append(mel)
          energy_list.append(energy)
          pitch_i_list.append(pitch_i)
          pitch_v_list.append(pitch_v)
          hubert_soft_list.append(hubert_soft)

      DIR.mkdir(parents=True, exist_ok=True)
      np_safesave(MELSPEC, np.concatenate(mel_list))
      np_safesave(ENERGY, np.concatenate(energy_list))
      np_safesave(PITCH_I, np.concatenate(pitch_i_list))
      np_safesave(PITCH_V, np.concatenate(pitch_v_list))
      np_safesave(HUBERT_SOFT, np.concatenate(hubert_soft_list))

if __name__ == "__main__":
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  START_HOP = 256
  AUDIO_HOP = 256 * START_HOP * 4  # 最初の 256 は HiFi-GAN の hop_size, 最後の 4 はバッチサイズ的な

  P.set_device(DEVICE)

  model = Attempt.VCModule.load_from_checkpoint(CKPT, map_location=DEVICE)
  model.eval()
  model.freeze()

  #

  prepare_audio()
  prepare_feats()
