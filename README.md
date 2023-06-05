This README is currently only available in Japanese.
Please [use machine translation](https://github-com.translate.goog/hoshi-vc/hoshi-vc/blob/main/README.md?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp) if necessary.

<!-- Note: Translations are welcome. (although this documentation is still incomplete...) -->

<br>
<div align="center">
  <h1>Hoshi-VC</h1>
  <p>A Personal Experiment in Real-Time Voice Conversion</p>
  <p>学習が高速なリアルタイム声質変換を作ってみる個人的な実験</p>
</div>
<br>
<br>

<!-- TODO: Add link to the demo and wandb. -->

## 開発のはじめかた

Linux や WSL2 での開発を想定しています。

```bash
# Install `asdf` and `pnpm` before you start if you haven't already.
# asdf: https://asdf-vm.com/guide/getting-started.html
# pnpm: https://pnpm.io/installation

# Clone this repository.
git clone https://github.com/hoshi-vc/hoshi-vc.git
cd hoshi-vc

# Install the required tools and packages.
asdf install
pdm  install -G :all
pnpm install

# Now you are ready to go!
source .venv/bin/activate
python engine/prepare.py
```

<!-- TODO: 環境構築の方法だけじゃなくて、学習の走らせ方などまで書きたい。 -->

## やってみたこと

### Attempt 01: 単純な音声の切り貼り

音声を切り貼りするだけでもうまく変換できるかもしれないので、試してみた。

[関連する Notebook](engine/attempt01.ipynb)

<!-- TODO: 生成結果の音声を貼る :: 動画形式にすれば GitHub のプレビューに埋め込める -->

### Attempt 02: 音程によらない表現をつかう

Attempt 01 では、音程があっていない音声を無理やりつなげているように見えた。

なので今度は、音程を調節してから切り貼りするようにしてみる。

→ 音程によらない表現をうまく作れなかったので、ひとまず後回しにした。

<!-- TODO: [関連する Notebook](engine/attempt02.ipynb) -->

### Attempt 03: FragmentVC をつかう

FragmentVC がどれくらいの性能なのか、公式実装で確認してみる。

→ なぜか学習がうまくいかなかったので、ひとまず後回しにした。

<!-- TODO: [関連する Notebook](engine/attempt03.ipynb) -->

### Attempt 04: 新しくモデルをつくってみる

FragmentVC がなぜかうまくいかなかったので、新しくモデルを作ってみた。

- CNN をベースにしてシンプルにモデルをつくる
- リファレンス音声の参照に cross-attention を使う（これは FragmentVC と似てる）
- 学習を容易にするために、ピッチや音量も特徴量としてつかう

→ FastSpeech2 をベースにして声質変換を試みたときと同様のペースでロスが低下してくれた。

<!-- - モデルの構造に FastSpeech2 の構造をつかってみる -->
<!-- - (ログ出力に attention map も追加する) -->

<!-- TODO: Write more details, results, observations, and conclusions. -->

## 参考にしたものなど

- [Faiss](https://github.com/facebookresearch/faiss) (efficient similarity search)
- [CLUB](https://arxiv.org/abs/2006.12013) (information bottleneck)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (phonetic feature extraction)
- [CREPE](https://arxiv.org/abs/1802.06182) (pitch estimation)
- [AdaSpeech](https://arxiv.org/abs/2103.00993) (conditional layer normalization)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) (audio waveform generation)
- [JVS corpus](https://arxiv.org/abs/1908.06248) (free multi-speaker voice corpus)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558), [FastPitch](https://arxiv.org/abs/2006.06873) (introduced me to the world of voice conversion)
- [FragmentVC](https://arxiv.org/abs/2010.14150) (inspired me to use a similarity search)

<!-- TODO: Comprehensive list of references. -->

## License

The code in this repository is licensed under the [Mozilla Public License 2.0](LICENSE).

Copyright 2023 Hoshi-VC Developer
