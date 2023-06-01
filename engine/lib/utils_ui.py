import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torch import Tensor

def plot_specgram(audio: Tensor, sr: int, title="Spectrogram", xlim=None):
  audio = audio.cpu().numpy()

  num_channels, _ = audio.shape

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(audio[c], Fs=sr)
    if num_channels > 1:
      axes[c].set_ylabel(f"Channel {c+1}")
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_spectrogram(specgram: Tensor, title=None, ylabel="freq_bin"):
  specgram = specgram.cpu()

  fig, axs = plt.subplots(1, 1)
  fig.set_size_inches(10, 3)

  axs.set_title(title or "Spectrogram")
  axs.set_ylabel(ylabel)
  axs.set_xlabel("frame")
  im = axs.imshow(specgram.transpose(0, 1), origin="lower", aspect="auto", interpolation="none")
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def play_audio(audio: Tensor, sr: int):
  audio = audio.cpu().numpy()

  if audio.ndim == 1: audio = audio[None, :]

  num_channels, _ = audio.shape
  if num_channels == 1:
    display(Audio(audio[0], rate=sr))
  elif num_channels == 2:
    display(Audio((audio[0], audio[1]), rate=sr))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")
