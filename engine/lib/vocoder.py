# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, as_tensor, no_grad
from torch.hub import download_url_to_file

from engine.hifi_gan.env import AttrDict
from engine.hifi_gan.models import Generator
from engine.lib.utils import Device

class HiFiGAN:
  def __init__(self, model: Any, config: Any):
    self._model = model
    self._config = config

  def __call__(self, mel: Tensor) -> tuple[Tensor, int]:
    model = self._model
    config = self._config
    device = self.device

    with no_grad():
      mel = mel.to(device)
      source_sr = 16000
      mel = convert_hop(mel, source_hop=320 * config.sampling_rate, target_hop=config.hop_size * source_sr, device=device)
      mel = mel.swapaxes(0, 1).unsqueeze(0).to(torch.float32)
      audio: Tensor = model(mel)
      audio = audio.squeeze()

    return audio, config.sampling_rate

  def to(self, device: Device):
    self._model = self._model.to(device)
    return self

  @property
  def device(self) -> Device:
    return next(self._model.parameters()).device

  @staticmethod
  def download_pretrained(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    download_url_to_file("https://drive.google.com/u/0/uc?id=1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e&export=download&confirm=t", str(dest / "config.json"))
    download_url_to_file("https://drive.google.com/u/0/uc?id=1O63eHZR9t1haCdRHQcEgMfMNxiOciSru&export=download&confirm=t", str(dest / "do_02500000"))
    download_url_to_file("https://drive.google.com/u/0/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download&confirm=t", str(dest / "g_02500000"))

  @staticmethod
  def load(model_dir: Path, ckpt: Optional[str] = None, download: bool = False):
    if download:
      if not model_dir.exists():
        HiFiGAN.download_pretrained(model_dir)

    if ckpt is None:
      ckpts = list(model_dir.glob("g_*"))
      if len(ckpts) != 1: raise ValueError(f"ckpt is required if model_dir is specified. ckpts: {ckpts}")
      ckpt = ckpts[0].name
    assert ckpt is not None

    config_path = model_dir / "config.json"
    model_path = model_dir / ckpt

    with open(config_path) as f:
      config = json.load(f)
    config = AttrDict(config)

    generator = Generator(config)

    state_dict_g = torch.load(model_path, map_location="cpu")
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    return HiFiGAN(generator, config)

def convert_hop(mel: Tensor, source_hop: int, target_hop: int, device: Device) -> Tensor:
  if source_hop == target_hop:
    return mel

  time_steps = np.arange(0, mel.shape[0], target_hop / source_hop)
  a = as_tensor((time_steps % 1.0).reshape((-1, 1))).to(device)
  mel1 = mel[(np.minimum(mel.shape[0] - 1, (time_steps).astype(int)))]
  mel2 = mel[(np.minimum(mel.shape[0] - 1, (time_steps + 1).astype(int)))]
  mel = (1 - a) * mel1 + a * mel2

  return mel
