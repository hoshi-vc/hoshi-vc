# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from os import fspath, path
from pathlib import Path
from typing import Literal, NamedTuple

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset

from engine.lib.utils import extract_zip, make_parents

# https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
URL = "https://drive.google.com/u/0/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt&export=download&confirm=t"
_CHECKSUM = {URL: "37180e2f87bd1a3e668d7c020378f77cebf61dd57d4d74c71eb0114f386a3999"}

JVSCategory = Literal["falset10", "nonpara30", "parallel100", "whisper10"]

class JVSEntry(NamedTuple):
  audio: Tensor
  sr: int
  name: str
  speaker_id: str
  category_id: JVSCategory
  utterance_id: str

class JVS(Dataset[JVSEntry]):
  speaker_ids = [f"jvs{i:03d}" for i in range(1, 101)]

  def __init__(
      self,
      root: str | Path,
      download: bool = False,
      url: str = URL,
      no_audio: bool = False,
  ) -> None:
    root = fspath(root)

    archive = path.join(root, "jvs_ver1.zip")
    data_dir = path.join(root, "jvs_ver1")

    if download:
      if not path.exists(data_dir):
        if not path.exists(archive):
          make_parents(archive)
          checksum = _CHECKSUM.get(url, None)
          download_url_to_file(url, archive, hash_prefix=checksum)
        make_parents(data_dir)
        extract_zip(archive, data_dir)
        os.remove(archive)

    if not path.exists(data_dir):
      raise RuntimeError(f"The path {data_dir} doesn't exist. "
                         "Please check the ``root`` path or set `download=True` to download it")

    self._path = str(Path(path.join(data_dir, "jvs_ver1")).resolve())

    files = sorted(str(p.relative_to(self._path)) for p in Path(self._path).glob("*/*/wav24kHz16bit/*.wav"))
    files = [Path(p) for p in files]
    entries: list[tuple[Path, str, str, str]] = []
    for filepath in files:
      (speaker_id, category_id, _, _) = filepath.parts
      utterance_id = filepath.stem
      entries.append((filepath, speaker_id, category_id, utterance_id))

    # 既知のミスがあるらしいので、パッチする
    # https://github.com/Hiroshiba/jvs_hiho/blob/cc72484e286a1d9a209118c8153a6a32e4c3c9ec/audio.bash

    patched: list[tuple[Path, str, str, str, str]] = []
    for (filepath, speaker_id, category_id, utterance_id) in entries:
      name = f"jvs/{speaker_id}/{category_id}/{utterance_id}"
      remove = False

      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_021": utterance_id = "VOICEACTRESS100_022"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_020": utterance_id = "VOICEACTRESS100_021"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_019": utterance_id = "VOICEACTRESS100_020"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_018": utterance_id = "VOICEACTRESS100_019"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_017": utterance_id = "VOICEACTRESS100_018"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_016": utterance_id = "VOICEACTRESS100_017"
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_015": utterance_id = "VOICEACTRESS100_016"
      if name == "jvs/jvs009/parallel100/VOICEACTRESS100_086": remove = True
      if name == "jvs/jvs009/parallel100/VOICEACTRESS100_095": remove = True
      if name == "jvs/jvs017/parallel100/VOICEACTRESS100_082": remove = True
      if name == "jvs/jvs018/parallel100/VOICEACTRESS100_072": remove = True
      if name == "jvs/jvs022/parallel100/VOICEACTRESS100_047": remove = True
      if name == "jvs/jvs024/parallel100/VOICEACTRESS100_088": remove = True
      if name == "jvs/jvs036/parallel100/VOICEACTRESS100_057": remove = True
      if name == "jvs/jvs038/parallel100/VOICEACTRESS100_006": remove = True
      if name == "jvs/jvs038/parallel100/VOICEACTRESS100_041": remove = True
      if name == "jvs/jvs043/parallel100/VOICEACTRESS100_085": remove = True
      if name == "jvs/jvs047/parallel100/VOICEACTRESS100_085": remove = True
      if name == "jvs/jvs048/parallel100/VOICEACTRESS100_043": remove = True
      if name == "jvs/jvs048/parallel100/VOICEACTRESS100_076": remove = True
      if name == "jvs/jvs051/parallel100/VOICEACTRESS100_025": remove = True
      if name == "jvs/jvs055/parallel100/VOICEACTRESS100_056": remove = True
      if name == "jvs/jvs055/parallel100/VOICEACTRESS100_076": remove = True
      if name == "jvs/jvs055/parallel100/VOICEACTRESS100_099": remove = True
      if name == "jvs/jvs058/parallel100/VOICEACTRESS100_014": remove = True
      if name == "jvs/jvs059/parallel100/VOICEACTRESS100_061": remove = True
      if name == "jvs/jvs059/parallel100/VOICEACTRESS100_064": remove = True
      if name == "jvs/jvs059/parallel100/VOICEACTRESS100_066": remove = True
      if name == "jvs/jvs059/parallel100/VOICEACTRESS100_074": remove = True
      if name == "jvs/jvs060/parallel100/VOICEACTRESS100_082": remove = True
      if name == "jvs/jvs074/parallel100/VOICEACTRESS100_062": remove = True
      if name == "jvs/jvs098/parallel100/VOICEACTRESS100_060": remove = True
      if name == "jvs/jvs098/parallel100/VOICEACTRESS100_099": remove = True

      prev_name = name
      name = f"jvs/{speaker_id}/{category_id}/{utterance_id}"

      # if prev_name != name: print(f"patched: {prev_name} -> {name}")
      # if remove: print(f"removed: {name}")
      if not remove: patched.append((filepath, name, speaker_id, category_id, utterance_id))

    self._entries = patched
    self._no_audio = no_audio

  def __getitem__(self, n: int) -> JVSEntry:
    (filepath, name, speaker_id, category_id, utterance_id) = self._entries[n]
    if self._no_audio:
      audio, sr = None, None
    else:
      audio, sr = torchaudio.load(path.join(self._path, filepath))
    return JVSEntry(audio, sr, name, speaker_id, category_id, utterance_id)

  def __len__(self) -> int:
    return len(self._entries)
