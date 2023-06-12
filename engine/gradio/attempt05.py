# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

# Inference UI for experiments
# https://gradio.app/docs/

print("Note: If the current source code is different from the one used for training, the result may be incorrect.")

import os

import faiss
import gradio as gr
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import engine.attempt05a as Attempt
from engine.dataset_feats import FeatureEntry4, IntraDomainDataset4
from engine.lib.utils import DATA_DIR, np_safesave
from engine.prepare import FEATS_DIR, Preparation

# TODO: index.reconstruct_batch を使って key を復元したいけど、なぜかうまくいかなかった。

CKPT = DATA_DIR / "attempt05/checkpoints/hopeful-capybara-3/2rmdgxq8/step=00027001-valid_spksim=0.4919-vcheat_spksim=0.5456.ckpt"

SAVE_DIR = DATA_DIR / "gradio" / CKPT.relative_to(DATA_DIR).with_suffix("")
MAX_LEN = 100000000000

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

  def __getitem__(self, index: int) -> FeatureEntry4:
    d, speaker_id, start = self.starts[index]

    # TODO: 面倒なので IntraDomainDataset4.load_entry を呼んでる
    return IntraDomainDataset4.load_entry(None, d, speaker_id, start, self.frames)

def prepare():
  for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, leave=False):
    SP_DIR = SAVE_DIR / speaker_id
    INDEX_FILE = SP_DIR / "faiss.index"
    KEY_FILE = SP_DIR / "keys.npy"
    VALUE_FILE = SP_DIR / "values.npy"

    if INDEX_FILE.exists() and KEY_FILE.exists() and VALUE_FILE.exists(): continue

    feat_dirs = [FEATS_DIR / "parallel100" / speaker_id]
    speaker_ids = [P.dataset.speaker_ids.index(speaker_id)]
    dataset = FeatureDataset(feat_dirs, speaker_ids, 256)
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    length = 0
    keys = []
    values = []
    for batch in tqdm(loader, ncols=0, desc=f"Loading {speaker_id}", leave=False):
      with torch.inference_mode():
        batch: FeatureEntry4 = model.transfer_batch_to_device(batch, model.device, 0)

        ref_energy = model.vc_model.forward_energy(batch.energy.float())
        ref_pitch = model.vc_model.forward_pitch(batch.pitch_i)
        # ref_phoneme = model.vc_model.forward_phoneme(batch.phoneme_i, batch.phoneme_v.float())
        ref_soft = model.vc_model.forward_w2v2(batch.soft.float())
        ref_key = model.vc_model.forward_key(ref_energy, ref_pitch, ref_soft)

        ref_mel = model.vc_model.forward_mel(batch.mel.float())
        ref_value = model.vc_model.forward_value(ref_energy, ref_pitch, ref_mel)

        # (batch, len, feat) -> (betch*len, feat)
        ref_key = ref_key.flatten(0, 1)
        ref_value = ref_value.flatten(0, 1)

        length += ref_key.shape[0]
        keys.append(ref_key.to(torch.float16).cpu().numpy())
        values.append(ref_value.to(torch.float16).cpu().numpy())

        if length > MAX_LEN: break

    keys = np.concatenate(keys[:MAX_LEN], axis=0)
    values = np.concatenate(values[:MAX_LEN], axis=0)

    SP_DIR.mkdir(parents=True, exist_ok=True)

    np_safesave(VALUE_FILE, values)
    np_safesave(KEY_FILE, keys)
    del values

    # index, index_infos = autofaiss.build_index(
    #     keys[:MAX_LEN // 10],
    #     save_on_disk=False,
    #     metric_type="ip",
    #     max_index_query_time_ms=1,
    #     max_index_memory_usage="10MB",
    #     min_nearest_neighbors_to_retrieve=16,
    # )

    index: faiss.IndexHNSWFlat = faiss.index_factory(keys.shape[1], "HNSW32,SQfp16", faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 400
    index.train(keys.astype(np.float32))
    index.add(keys.astype(np.float32))
    assert index.is_trained
    del keys

    faiss.write_index(index, str(INDEX_FILE) + ".tmp")
    os.replace(str(INDEX_FILE) + ".tmp", INDEX_FILE)

def convert(audio, sr, tgt_speaker_id, pitch_scale, pitch_shift, query_prep):
  energy = P.extract_energy(audio, sr)
  # phoneme_i, phoneme_v = P.extract_phoneme(audio, sr)
  soft = P.extract_hubert_soft(audio, sr)
  pitch_i, _ = P.extract_pitch(audio, sr)

  with torch.inference_mode():
    energy = energy.to(model.device, torch.float32).unsqueeze(0)
    # phoneme_i = phoneme_i.to(model.device, torch.int64).unsqueeze(0)
    # phoneme_v = phoneme_v.to(model.device, torch.float32).unsqueeze(0)
    soft = soft.to(model.device, torch.float32).unsqueeze(0)
    pitch_i = pitch_i.to(model.device, torch.float32).unsqueeze(0)
    pitch_i = torch.clamp(pitch_i * pitch_scale + pitch_shift, 0, 360 - 1)
    pitch_i = pitch_i.to(torch.int64)

    src_energy = model.vc_model.forward_energy(energy)
    src_pitch = model.vc_model.forward_pitch(pitch_i)
    # src_phoneme = model.vc_model.forward_phoneme(phoneme_i, phoneme_v)
    src_soft = model.vc_model.forward_w2v2(soft)
    src_key = model.vc_model.forward_key(src_energy, src_pitch, src_soft)

    index = faiss.read_index(str(SAVE_DIR / tgt_speaker_id / "faiss.index"))
    keys = np.load(str(SAVE_DIR / tgt_speaker_id / "keys.npy"), mmap_mode="r")
    values = np.load(str(SAVE_DIR / tgt_speaker_id / "values.npy"), mmap_mode="r")

    _, ref_indices = index.search(src_key.squeeze(0).cpu().numpy(), 64)

    ref_keys = torch.as_tensor(keys[ref_indices]).to(model.device, torch.float32)
    ref_values = torch.as_tensor(values[ref_indices]).to(model.device, torch.float32)

    if query_prep == "None": pass
    elif query_prep == "Top1": src_key = ref_keys[:, 0].unsqueeze(0)
    elif query_prep == "Top64-Mean": src_key = ref_keys.mean(dim=1).unsqueeze(0)
    else: raise NotImplementedError

    # ref_key = index.reconstruct_batch(ref_indices)
    ref_key = ref_keys.flatten(0, 1).unsqueeze(0)
    ref_value = ref_values.flatten(0, 1).unsqueeze(0)

    tgt_value, _ = model.vc_model.lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = model.vc_model.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))
    tgt_audio = model.vocoder(tgt_mel.transpose(1, 2)).squeeze(1)

    return tgt_audio, 22050

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

P = Preparation(DEVICE)

Attempt.P = P  # TODO: もっといい方法ない？
model = Attempt.VCModule.load_from_checkpoint(CKPT, map_location=DEVICE)
model.eval()
model.freeze()

prepare()

# %%

# item = P.dataset[500]
# audio, sr = item.audio[0], item.sr
# print(item.name)
# play_audio(audio, sr)
# tgt_audio, tgt_sr = convert(audio, sr, "jvs001")
# play_audio(tgt_audio.cpu().numpy(), tgt_sr)

title = "Hoshi-VC Conversion Demo"

description = """
<div style="text-align: center">
  A Personal Experiment in Real-Time Voice Conversion
</div>
<p style="margin: 3rem 1rem 0rem">
  Upload an audio file or record using the microphone. First conversion may take a while.
</p>
<p style="margin: 0rem 1rem 3rem">
  NOTE: This is a provisional UI for development.
</p>
"""

article = """
<p style="text-align: center">
  Project Page: <a href="https://github.com/hoshi-vc/hoshi-vc">GitHub</a>
</p>
"""

examples = [
    ["jvs001", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs002", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs002/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs003", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs003/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs004", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs004/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs005", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs006", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs006/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs007", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs007/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs008", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs008/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs009", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
    ["jvs010", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs010/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", None, 0.0],
]
speaker_ids = P.dataset.speaker_ids

# with gr.Blocks(title="Hoshi-VC") as app:
#   with gr.Row():
#     with gr.Column():
#       with gr.Tabs():
#         with gr.Tab("Upload"):
#           upload_audio = gr.Audio(source="upload", type="numpy")
#         with gr.Tab("Record"):
#           mic_audio = gr.Audio(source="microphone", type="numpy")
#       speaker = gr.Dropdown(label="Target Speaker", choices=speaker_ids, value=speaker_ids[0], interactive=True)
#     with gr.Column():
#       output = gr.Audio(label="Converted Speech", type="numpy", interactive=False)

def process(speaker, upload_audio, mic_audio, pitch_shift, query_prep):
  # audio = tuple (sample_rate, frames) or (sample_rate, (frames, channels))
  if mic_audio is not None:
    sr, audio = mic_audio
  elif upload_audio is not None:
    sr, audio = upload_audio
  else:
    return (22050, np.zeros(0).astype(np.int16))

  if audio.dtype == np.int16:
    audio = torch.as_tensor(audio, dtype=torch.float32, device=model.device) / 32768.0
  elif audio.dtype == np.int32:
    audio = torch.as_tensor(audio, dtype=torch.float32, device=model.device) / 2147483648.0
  elif audio.dtype == np.float16 or audio.dtype == np.float32 or audio.dtype == np.float64:
    audio = torch.as_tensor(audio, dtype=torch.float32, device=model.device)
  else:
    raise ValueError("Unsupported dtype")

  tgt_audio, tgt_sr = convert(audio, sr, speaker, 1.0, pitch_shift, query_prep)

  tgt_audio = tgt_audio.unsqueeze(0).cpu().numpy() * 32768.0

  return tgt_sr, tgt_audio.astype(np.int16)

app = gr.Interface(
    fn=process,
    inputs=[
        gr.Dropdown(label="Target Speaker", choices=speaker_ids, value=speaker_ids[0]),
        gr.Audio(label="Upload Speech", source="upload", type="numpy"),
        gr.Audio(label="Record Speech", source="microphone", type="numpy"),
        gr.Slider(label="Pitch Shift", minimum=-100, maximum=100, step=1.0, value=0.0),
        gr.Radio(label="Query Preprocess", choices=["None", "Top1", "Top64-Mean"], value="None"),
    ],
    outputs=[
        gr.Audio(label="Converted Speech", type="numpy"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
)

app.queue(1)
app.launch()
