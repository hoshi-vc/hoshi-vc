# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).
#
# ## Attempt 01: 単純な音声の切り貼り
#
# 変換先の音声が十分にあるなら、変換元の声の発音に近い変換先話者の音声を切り貼りするだけでもそれなりにうまく変換できるかもしれないと思ったので、試してみる。

# %%

import numpy as np
import torch

from engine.lib.utils import NPArray
from engine.lib.utils_ui import play_audio, plot_spectrogram
from engine.prepare import Preparation

device = "cuda"

P = Preparation(device)

P.prepare_feats()
P.prepare_faiss()

# %%

index = P.get_index("jvs001")
target_mel = P.get_mel("jvs001")

item = P.dataset[1000]
print(item.name)

audio, sr = item.audio[0], item.sr
mel = P.extract_melspec(audio, sr)
keys = P.extract_wav2vec2(audio, sr).cpu().numpy()

top_k = 16

hat = []
for i in range(len(keys)):
  D, I = index.search(keys[None, i], top_k)
  items: list[NPArray] = []
  for j in range(top_k):
    items.append(target_mel[I[0][j]])
  hat.append(np.mean(np.stack(items), axis=0))
hat = torch.as_tensor(np.vstack(hat))

audio_hat, sr_hat = P.vocoder(hat)
plot_spectrogram(mel)
plot_spectrogram(hat)
play_audio(audio, sr)
play_audio(audio_hat, sr_hat)

# %%
