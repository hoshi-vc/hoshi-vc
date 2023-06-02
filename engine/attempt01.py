# Faiss Unofficial Guide: https://www.pinecone.io/learn/faiss-tutorial/

# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).
#
# ## Attempt 01: 単純な音声の切り貼り
#
# 変換先の音声が十分にあるなら、変換元の声の発音に近い変換先話者の音声を切り貼りするだけでもそれなりにうまく変換できるかもしれないと思ったので、試してみる。

# %%

import numpy as np
import torch
from tqdm import tqdm

from engine.lib.utils import NPArray
from engine.lib.utils_ui import play_audio, plot_spectrogram
from engine.preparation import Preparation

device = "cuda"

P = Preparation(device)

P.prepare_kv_pairs()
P.prepare_faiss()

# %%

index = P.get_index("jvs001")
values = P.get_values("jvs001")

item = P.dataset[1000]
print(item.name)

audio, sr = item.audio[0], item.sr
mel = P.extract_melspec(audio, sr)
keys = P.extract_wav2vec2(audio, sr).cpu().numpy()

top_k = 128

hat = []
for i in range(len(keys)):
  D, I = index.search(keys[None, i], top_k)
  items: list[NPArray] = []
  for j in range(top_k):
    items.append(values[I[0][j]])
  hat.append(np.mean(np.stack(items), axis=0))
hat = torch.as_tensor(np.vstack(hat))

audio_hat, sr_hat = P.vocoder(hat)
plot_spectrogram(mel)
plot_spectrogram(hat)
play_audio(audio, sr)
play_audio(audio_hat, sr_hat)

# %%
