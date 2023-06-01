# %%
import numpy as np
import torch
from torchaudio.functional import resample
from tqdm import tqdm

from engine.lib.dataset_jvs import JVS
from engine.lib.extract import Wav2Vec2, extract_melspec
from engine.lib.utils import DATA_DIR
from engine.lib.utils_ui import play_audio, plot_spectrogram
from engine.lib.vocoder import HiFiGAN

DIR = DATA_DIR / "attempt01"
DIR.mkdir(parents=True, exist_ok=True)

dataset = JVS(DATA_DIR / "datasets", download=True)
vocoder = HiFiGAN.load(DATA_DIR / "vocoder", download=True)
extract_wav2vec2 = Wav2Vec2.load("facebook/wav2vec2-base")

item = dataset[0]
audio, sr = item.audio[0], item.sr
wav2vec2 = extract_wav2vec2(audio, sr)
mel = extract_melspec(audio, sr)
audio_hat, sr_hat = vocoder(mel)
mel_hat = extract_melspec(audio_hat, sr_hat)[:mel.shape[0]]

print("original audio")
play_audio(audio.unsqueeze(0), sr)
print("reconstructed by vocoder")
play_audio(audio_hat.unsqueeze(0), sr_hat)
plot_spectrogram((mel - mel_hat)**2, "Difference")
