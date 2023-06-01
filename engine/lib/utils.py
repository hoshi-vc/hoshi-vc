import os
import zipfile
from os import path
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

from numpy.typing import NDArray
from torch import device as TorchDevice
from torch.utils.data import Dataset
from tqdm import tqdm

T = TypeVar("T")

NPArray = NDArray[Any]
Device = Union["TorchDevice", str]

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def extract_zip(src: Path | str, dest: Path | str) -> list[str]:
  with zipfile.ZipFile(src, "r") as zf:
    for file in tqdm(zf.namelist(), ncols=0):
      zf.extract(file, dest)

def make_parents(file: str | Path):
  os.makedirs(path.dirname(file), exist_ok=True)

class FilteredDataset(Dataset[T]):
  def __init__(self, base: Dataset[T], fn: Callable[[T], bool]) -> None:
    self._base = base
    self._fn = fn
    self._indices = [i for i, x in tqdm(enumerate(base), total=len(base), desc="FilteredDataset", ncols=0) if fn(x)]

  def __getitem__(self, n: int) -> T:
    return self._base[self._indices[n]]

  def __len__(self) -> int:
    return len(self._indices)
