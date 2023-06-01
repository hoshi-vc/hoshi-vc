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

    self._path = path.join(data_dir, "jvs_ver1")
    self._walker = sorted(str(p.relative_to(self._path)) for p in Path(self._path).glob("*/*/wav24kHz16bit/*.wav"))
    self._walker = [Path(p) for p in self._walker]

  def __getitem__(self, n: int) -> JVSEntry:
    filepath = self._walker[n]
    (speaker_id, category_id, _, utterance_id) = filepath.parts
    audio, sr = torchaudio.load(path.join(self._path, filepath))
    return JVSEntry(audio, sr, speaker_id, category_id, utterance_id)

  def __len__(self) -> int:
    return len(self._walker)
