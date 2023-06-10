Please [use machine translation](https://github-com.translate.goog/hoshi-vc/hoshi-vc/blob/main/README.md?_x_tr_sl=en&_x_tr_tl=ja&_x_tr_hl=en&_x_tr_pto=wapp) if necessary.<br>
でも、[日本語版ならここに](README_JA.md)あります。

<!-- Note: Translations are welcome. -->

<br>
<div align="center">
  <h1>Hoshi-VC</h1>
  <p>A Personal Experiment in Real-Time Any-to-Many Voice Conversion</p>
</div>
<br>
<br>

## Getting Started with Development

It is intended for development on Linux and WSL2.

```bash
# Install `asdf` and `pnpm` before you start if you haven't already.
# asdf: https://asdf-vm.com/guide/getting-started.html
# pnpm: https://pnpm.io/installation

# Clone this repository.
git clone https://github.com/hoshi-vc/hoshi-vc.git
cd hoshi-vc

# Install the necessary tools and packages.
asdf install
pdm  install -G :all
pnpm install

# Now you are ready to go!
source .venv/bin/activate
python engine/prepare.py
```

Note: The `requirements.txt` is kept up to date, so you can use that.

## Progress

This project is currently under active development.

Please note that the source code is updated constantly.

Feel free to star this repo if you are interested in updates.

- [x] Prepare the preprocessing pipeline
- [x] Create a base conversion model
- [x] Experiment with different architectures
  - [x] Search for related speech fragments with cross-attention
  - [x] Make the latent space pitch independent with CLUB
  - [x] Train model and vocoder together
  - [x] Use FastSpeech's Feed Forward Transformer
  - [x] Use ACGAN to increase speaker similarity
  - [ ] Make latent space speaker independent if necessary
  - [ ] Use AdaSpeech's conditional layer normalization
  - [ ] And other improvements...
- [x] Create a temporary inference UI for development
  - [x] Write an inference script using Faiss
  - [x] Build a temporary UI with Gradio
  - [x] Make it faster than real time
- [ ] Create a real-time conversion client
- [ ] Make it easy for everyone to use

## What I Tried

### Attempt 01: Simple voice cut and paste

Just cutting and pasting the audio might work, so I tried it.

[Related Notebook](engine/attempt01.ipynb)

### Attempt 02: Use pitch-independent embedding

In Attempt 01, it seemed that audio fragments with out-of-tune pitches were combined to generate audio.

So, I tried to adjust the pitch before cutting and pasting.

→ Since I could not successfully create a pitch-independent embedding, I postponed it for the moment.

### Attempt 03: Use FragmentVC

Check the performance of FragmentVC by using the official implementation.

→ I postponed this for now because for some reason the learning didn't go well.

### Attempt 04: Create a new model

FragmentVC didn't work well for some reason, so I decided to create a new model.

- A simple model based on CNN
- Use cross-attention to cut and paste audio (similar to FragmentVC)
- Also use pitch and volume as features to make learning easier.

→ The loss decreased at the same rate as when I tried voice conversion based on FastSpeech2.

### Attempt 05: Combine model and vocoder

Next, I thought about using GAN, but since the pre-trained weights of the HiFi-GAN discriminator are publicly available, I decided to do joint fine-tuning.

According to [JETS](https://arxiv.org/pdf/2203.16852.pdf) and [ESPnet2-TTS](https://arxiv.org/pdf/2110.07840.pdf), a jointly trained model performs better than a fine-tuned vocoder.

Also, some additional metrics should be added, since melspectrogram l1 loss is no longer suitable for performance evaluation when GAN is integrated into the training.

- [MOSNet](https://github.com/aliutkus/speechmetrics#mosnet-absolutemosnet-or-mosnet) : mean opinion score
- [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) : speaker similarity score

## References

- [Faiss](https://github.com/facebookresearch/faiss) (efficient similarity search)
- [CLUB](https://arxiv.org/abs/2006.12013) (information bottleneck)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (phonetic feature extraction)
- [CREPE](https://arxiv.org/abs/1802.06182) (pitch estimation)
- [AdaSpeech](https://arxiv.org/abs/2103.00993) (conditional layer normalization)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) (audio waveform generation)
- [JVS corpus](https://arxiv.org/abs/1908.06248) (free multi-speaker voice corpus)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558), [FastPitch](https://arxiv.org/abs/2006.06873) (introduced me to the world of voice conversion)
- [FragmentVC](https://arxiv.org/abs/2010.14150) (inspired me to use a similarity search)

## License

The code in this repository is licensed under the [Mozilla Public License 2.0](LICENSE).

Copyright 2023 Hoshi-VC Developer
