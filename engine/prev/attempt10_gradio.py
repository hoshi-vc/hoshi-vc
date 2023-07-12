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
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import engine.prev.attempt10 as Attempt
from engine.lib.pyworld import pyworld_vc
from engine.lib.utils import np_safesave
from engine.prev.attempt10_dataset import Dataset10, Feats10
from engine.singleton import DATA_DIR, FEATS_DIR, P

# TODO: index.reconstruct_batch を使って key を復元したいけど、なぜかうまくいかなかった。

# CKPT = DATA_DIR / "attempt07/checkpoints/still-sponge-23/xe0uvvce/last.ckpt"
# CKPT = DATA_DIR / "attempt07/checkpoints/devout-leaf-28/n5yobotl/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/dulcet-plant-77/qmnaevyy/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/driven-frost-81/q4zejkzl/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/rural-fog-97/m9upk9dm/last.ckpt"  # 08ca
# CKPT = DATA_DIR / "attempt08/checkpoints/quiet-mountain-101/4w4378y0/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/earnest-wood-107/rsvomzlu/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/graceful-disco-108/29w53sxj/last.ckpt"
# CKPT = DATA_DIR / "attempt08/checkpoints/driven-butterfly-113/33a18oza/last.ckpt"  # 08e
# CKPT = DATA_DIR / "attempt08/checkpoints/dutiful-forest-130/83lxd9jt/last.ckpt" # 08f
# CKPT = DATA_DIR / "attempt08/checkpoints/icy-puddle-139/2qsisbet/last.ckpt" # 08, low ref
# CKPT = DATA_DIR / "attempt08/checkpoints/dark-fire-140/plmh06q1/last.ckpt"  # 08h
# CKPT = DATA_DIR / "attempt08/checkpoints/golden-sun-141/m44ug75k/last.ckpt"  # 08h
CKPT = DATA_DIR / "attempt08/checkpoints/fresh-paper-144/6keqvh6s/last.ckpt"  # 08ca

SAVE_DIR = DATA_DIR / "gradio" / CKPT.relative_to(DATA_DIR).with_suffix("")

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

  def __getitem__(self, index: int) -> Feats10:
    d, speaker_id, start = self.starts[index]

    # TODO: 面倒なので直接呼んでる
    return Feats10.load(None, d, speaker_id, start, self.frames)

def prepare():
  for speaker_id in tqdm(P.dataset.speaker_ids, ncols=0, leave=False):
    SP_DIR = SAVE_DIR / speaker_id
    INDEX_FILE = SP_DIR / "faiss.index"
    KEY_FILE = SP_DIR / "keys.npy"
    VALUE_FILE = SP_DIR / "values.npy"
    SPKEMBS_FILE = SP_DIR / "spkembs.npy"

    if INDEX_FILE.exists() and KEY_FILE.exists() and VALUE_FILE.exists() and SPKEMBS_FILE.exists(): continue

    feat_dirs = [FEATS_DIR / "parallel100" / speaker_id]
    speaker_ids = [P.dataset.speaker_ids.index(speaker_id)]
    dataset = FeatureDataset(feat_dirs, speaker_ids, 256)
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    length = 0
    keys = []
    values = []
    spkembs = []
    for batch in tqdm(loader, ncols=0, desc=f"Loading {speaker_id}", leave=False):
      with torch.inference_mode():
        batch: Feats10 = model.transfer_batch_to_device(batch, model.device, 0)

        ref_energy = model.vc_model.forward_energy(batch.energy.float())
        ref_pitch = model.vc_model.forward_pitch(batch.pitch_i)
        # ref_phoneme = model.vc_model.forward_phoneme(batch.phoneme_i, batch.phoneme_v.float())
        ref_soft = model.vc_model.forward_w2v2(batch.soft.float())
        ref_key = model.vc_model.forward_key(ref_energy, ref_pitch, ref_soft)

        ref_mel = model.vc_model.forward_mel(batch.mel.float())
        ref_value = model.vc_model.forward_value(ref_energy, ref_pitch, ref_mel)

        # (batch, len, feat) -> (betch*len, feat)
        n_batch, n_len, _ = ref_key.shape
        ref_key = ref_key.flatten(0, 1)
        ref_value = ref_value.flatten(0, 1)

        spkemb = P.spkemb(batch.audio, 22050)
        assert len(spkemb.shape) == 2 and spkemb.shape[0] == n_batch
        spkemb = spkemb.unsqueeze(1).expand(-1, n_len, -1).flatten(0, 1)

        length += ref_key.shape[0]
        keys.append(ref_key.to(torch.float16).cpu().numpy())
        values.append(ref_value.to(torch.float16).cpu().numpy())
        spkembs.append(spkemb.to(torch.float16).cpu().numpy())

    keys = np.concatenate(keys, axis=0)
    values = np.concatenate(values, axis=0)
    spkembs = np.concatenate(spkembs, axis=0)

    SP_DIR.mkdir(parents=True, exist_ok=True)

    np_safesave(VALUE_FILE, values)
    np_safesave(KEY_FILE, keys)
    np_safesave(SPKEMBS_FILE, spkembs)
    del values

    # index, index_infos = autofaiss.build_index(
    #     keys,
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

def convert(audio, sr, tgt_speaker_id, pitch_scale, pitch_shift, ex_key, ex_val, topk, query_prep):
  energy = P.extract_energy(audio, sr)
  # phoneme_i, phoneme_v = P.extract_phoneme(audio, sr)
  mel = P.extract_melspec(audio, sr)
  soft = P.extract_hubert_soft(audio, sr)
  pitch_i, pitch_v = P.extract_pitch(audio, sr)

  print(f"pitch: {pitch_i[pitch_v[:, 0] > 0.5, 0].float().mean().item()}")

  with torch.inference_mode():
    energy = energy.to(model.device, torch.float32).unsqueeze(0)
    # energy = torch.round(energy * 2) / 2
    # phoneme_i = phoneme_i.to(model.device, torch.int64).unsqueeze(0)
    # phoneme_v = phoneme_v.to(model.device, torch.float32).unsqueeze(0)
    mel = mel.to(model.device, torch.float32).unsqueeze(0)
    soft = soft.to(model.device, torch.float32).unsqueeze(0)
    pitch_i = pitch_i.to(model.device, torch.float32).unsqueeze(0)

    pitch_i = pitch_i * pitch_scale + pitch_shift
    # pitch_i = torch.round(pitch_i / 16) * 16
    pitch_i = torch.clamp(pitch_i, 0, 360 - 1)
    pitch_i = pitch_i.to(torch.int64)

    src_energy = model.vc_model.forward_energy(energy)
    src_pitch = model.vc_model.forward_pitch(pitch_i)
    src_mel = model.vc_model.forward_mel(mel)
    # src_phoneme = model.vc_model.forward_phoneme(phoneme_i, phoneme_v)
    src_soft = model.vc_model.forward_w2v2(soft)
    src_key = model.vc_model.forward_key(src_energy, src_pitch, src_soft)
    src_value = model.vc_model.forward_value(src_energy, src_pitch, src_mel)  # !!: 今まで間違えて src_soft を渡してた
    src_spkemb = P.spkemb(audio, sr)

    index = faiss.read_index(str(SAVE_DIR / tgt_speaker_id / "faiss.index"))
    keys = np.load(str(SAVE_DIR / tgt_speaker_id / "keys.npy"), mmap_mode="r")
    values = np.load(str(SAVE_DIR / tgt_speaker_id / "values.npy"), mmap_mode="r")
    spkembs = np.load(str(SAVE_DIR / tgt_speaker_id / "spkembs.npy"), mmap_mode="r")

    _, ref_indices = index.search(src_key.squeeze(0).cpu().numpy(), topk)

    ref_keys = torch.as_tensor(keys[ref_indices]).to(model.device, torch.float32)
    ref_values = torch.as_tensor(values[ref_indices]).to(model.device, torch.float32)
    ref_spkembs = torch.as_tensor(spkembs[ref_indices]).to(model.device, torch.float32)

    # print(src_key[:10, :4])

    if query_prep == "None": pass
    elif query_prep == "Top1": src_key = ref_keys[:, 0].unsqueeze(0)
    elif query_prep == "TopK-Mean": src_key = ref_keys.mean(dim=1).unsqueeze(0)
    else: raise NotImplementedError

    # print(src_key[:10, :4])

    if ex_key != 1.0:
      src_keys = src_key.permute(1, 0, 2)
      ref_keys = src_keys + (ref_keys - src_keys) * ex_key

    if ex_val != 1.0:
      src_values = src_value.permute(1, 0, 2)
      ref_values = src_values + (ref_values - src_values) * ex_val

    # ref_indices = np.arange(len(src_key) * 1024).reshape(len(src_key), -1)

    # ref_key = index.reconstruct_batch(ref_indices)
    ref_key = ref_keys.flatten(0, 1).unsqueeze(0)
    ref_value = ref_values.flatten(0, 1).unsqueeze(0)
    ref_spkemb = ref_spkembs.flatten(0, 1).unsqueeze(0)

    tgt_value, _ = model.vc_model.forward_lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = model.vc_model.forward_decode(tgt_value, src_energy, src_pitch)
    tgt_audio = model.vocoder(tgt_mel.transpose(1, 2)).squeeze(1)

    # print some data
    # import matplotlib.pyplot as plt
    # from engine.lib.utils_ui import plot_spectrogram
    # plt.switch_backend('TkAgg')
    # plt.hist(torch.cosine_similarity(src_spkemb, ref_spkemb, dim=-1).squeeze(0).cpu().numpy(), bins=100)
    # plt.show()
    # plt.imshow(tgt_mel.squeeze(0).cpu().numpy().T, aspect="auto", origin="lower")
    # plt.show()
    # plt.hist(pitch_i.view(-1).cpu().numpy(), 100, (0, 360))
    # plt.show()

    return tgt_audio, 22050

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

P.set_device(DEVICE)

model = Attempt.VCModule.load_from_checkpoint(CKPT, map_location=DEVICE)
model.eval()
model.freeze()

# %%

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
    ["jvs001", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs002", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs002/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs003", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs003/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs004", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs004/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs005", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs006", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs006/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs007", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs007/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs008", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs008/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs009", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
    ["jvs010", DATA_DIR / "datasets/jvs_ver1/jvs_ver1/jvs010/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"],
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

def process(speaker, upload_audio, mic_audio, pre_ps, pre_fs, pitch_scale, pitch_shift, gain, ex_key, ex_val, topk, query_prep):
  # audio = tuple (sample_rate, frames) or (sample_rate, (frames, channels))
  if mic_audio is not None:
    sr, audio = mic_audio
  elif upload_audio is not None:
    sr, audio = upload_audio
  else:
    return (22050, np.zeros(0).astype(np.int16))

  if audio.dtype == np.int16:
    audio = audio / 32768.0
  elif audio.dtype == np.int32:
    audio = audio / 2147483648.0
  elif audio.dtype == np.float16 or audio.dtype == np.float32 or audio.dtype == np.float64:
    audio = audio
  else:
    raise ValueError("Unsupported dtype")

  audio = torch.as_tensor(audio, dtype=torch.float32, device=model.device)
  audio, sr = torchaudio.functional.resample(audio, sr, 22050), 22050

  if pre_ps != 0 or pre_fs != 0:
    audio_prep = audio.cpu().numpy()
    audio_prep = pyworld_vc(audio_prep, sr, pre_ps, pre_fs)
    audio_prep = torch.as_tensor(audio_prep, dtype=torch.float32, device=model.device)
  else:
    audio_prep = audio

  audio_norm = P.normalize_audio(audio_prep, sr, trim=False) * gain

  tgt_audio_norm, tgt_sr = convert(audio_norm, sr, speaker, pitch_scale, pitch_shift, ex_key, ex_val, topk, query_prep)

  tgt_audio = tgt_audio_norm * (audio**2).mean().sqrt() / (tgt_audio_norm**2).mean().sqrt()

  tgt_audio = tgt_audio.unsqueeze(0).cpu().numpy() * 32768.0
  audio_prep = audio_prep.cpu().numpy() * 32768.0

  return (tgt_sr, audio_prep.astype(np.int16)), (tgt_sr, tgt_audio.astype(np.int16))

app = gr.Interface(
    fn=process,
    inputs=[
        gr.Dropdown(label="Target Speaker", choices=speaker_ids, value=speaker_ids[0]),
        gr.Audio(label="Upload Speech", source="upload", type="numpy"),
        gr.Audio(label="Record Speech", source="microphone", type="numpy"),
        gr.Slider(label="Preprocess: Pitch Shift", minimum=-100, maximum=100, step=1.0, value=0.0),
        gr.Slider(label="Preprocess: Formant Shift", minimum=-100, maximum=100, step=1.0, value=0.0),
        gr.Slider(label="Pitch Scale", minimum=-0.0, maximum=2.0, step=0.01, value=1.0),
        gr.Slider(label="Pitch Shift", minimum=-100, maximum=100, step=1.0, value=0.0),
        gr.Slider(label="Gain", minimum=0.1, maximum=10, step=0.1, value=1.0),
        gr.Slider(label="Exaggeration (Key)", minimum=-4.0, maximum=4.0, step=0.1, value=1.0),
        gr.Slider(label="Exaggeration (Value)", minimum=-4.0, maximum=4.0, step=0.1, value=1.0),
        gr.Slider(label="TopK", minimum=1, maximum=64, step=1, value=64),
        gr.Radio(label="Query Preprocess", choices=["None", "Top1", "TopK-Mean"], value="None"),
    ],
    outputs=[
        gr.Audio(label="Preprocessed", type="numpy"),
        gr.Audio(label="Converted Speech", type="numpy"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
)

app.queue(1)
app.launch()
