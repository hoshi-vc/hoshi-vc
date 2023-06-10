English version: [README.md](README.md)

<!-- Note: Translations are welcome. -->

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
# インストールしていない場合は `asdf` と `pnpm` を入れておいてください。
# asdf: https://asdf-vm.com/guide/getting-started.html
# pnpm: https://pnpm.io/installation

# このレポジトリをクローンします。
git clone https://github.com/hoshi-vc/hoshi-vc.git
cd hoshi-vc

# 必要なツールなどをインストールします。
asdf install
pdm  install -G :all
pnpm install

# これで準備完了です！
source .venv/bin/activate
python engine/prepare.py
```

Note: `requirements.txt` は最新に保たれているので、それを使ってもいいです。

<!-- TODO: 環境構築の方法だけじゃなくて、学習の走らせ方などまで書きたい。 -->

## 開発の進捗

- [x] 前処理のパイプラインを用意する
- [x] 基本的な変換モデルを作る
- [x] 色々なモデル構造で実験する
  - [x] クロスアテンションで関連する音声の部分をかき集める
  - [x] CLUB で音程によらない潜在空間を作る
  - [x] モデルとボコーダを同時に学習する
  - [x] FastSpeech の Feed Forward Transformer を使う
  - [x] ACGAN で話者の類似性を高める
  - [ ] 潜在空間を話者非依存にする（必要なら）
  - [ ] AdaSpeech のやつを使う
  - [ ] 他にもいろいろ試す...
- [x] 開発のための仮の変換画面を作る
  - [x] Faiss を使って推論スクリプトを書く
  - [x] Gradio で仮の UI を作る
  - [x] リアルタイムより速くする
- [ ] リアルタイム変換クライアントを作る
- [ ] みんなが使いやすいようにする

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

### Attempt 05: モデルと vocoder を組み合わせる

次は GAN を入れようと思ったけど、せっかくなので joint fine-tuning してみる。

[JETS](https://arxiv.org/pdf/2203.16852.pdf) や [ESPnet2-TTS](https://arxiv.org/pdf/2110.07840.pdf) によると、 fine-tuned vocoder よりも joint fine-tuning の方がいいらしい。

学習に GAN を取り入れると melspectrogram l1 loss が性能評価に適さなくなるので、指標を増やす。

- [MOSNet](https://github.com/aliutkus/speechmetrics#mosnet-absolutemosnet-or-mosnet) : 人間の主観評価を推測する
- [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) : 話者がどれだけ似ているかを評価する

<!-- - モデルの構造に FastSpeech2 の構造をつかってみる -->
<!-- - (ログ出力に attention map も追加する) -->

<!-- TODO: Write more details, results, observations, and conclusions. -->

## 参考にしたものなど

- [Faiss](https://github.com/facebookresearch/faiss) （高速なベクトル検索）
- [CLUB](https://arxiv.org/abs/2006.12013) （情報の分離）
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) （音素などの抽出）
- [CREPE](https://arxiv.org/abs/1802.06182) （音の高さの推定）
- [AdaSpeech](https://arxiv.org/abs/2103.00993) （ conditional layer normalization ）
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) （音声波形の生成）
- [JVS corpus](https://arxiv.org/abs/1908.06248) （複数話者の発話音声のコーパス）
- [FastSpeech 2](https://arxiv.org/abs/2006.04558), [FastPitch](https://arxiv.org/abs/2006.06873) （声質変換を始めたきっかけ）
- [FragmentVC](https://arxiv.org/abs/2010.14150) （ベクトル検索を使うというアイデアをもらった）

<!-- TODO: Comprehensive list of references. -->

## ライセンスについて

このレポジトリのコードは [Mozilla Public License 2.0](LICENSE) のもとで公開しています。

Copyright 2023 Hoshi-VC Developer
