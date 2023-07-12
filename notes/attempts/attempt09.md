This document is currently only available in Japanese.
Please [use machine translation](https://github-com.translate.goog/hoshi-vc/hoshi-vc?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp) if necessary.

# 09 : StarGAN

## 背景

- 同一話者の音声再合成は、十分な音質で実現できるようになった
- 話者をまたいだ変換では、それほどうまく行っていない
  - ピッチは保持されてる
  - 発音はそれなりに聞き取れる
  - 声質は変化している
  - でもターゲット話者の声にはならない
  - `jvs001 -> jvs002` とかでは無理して高音を出してる感じの声になる
- 話者性の変換を直接学習させていないことが原因かなと思う
  - 今はボトルネックだけに頼ってる
  - それだけで話者性は除去しきれないのかなと思う
  - 一緒に発音情報まで除去されてそう
- しかし、パラレル音声は用意できない
  - 学習済み VC モデルでパラレル音声を生成するのもうまくいかなかった
  - フォルマントやピッチのシフトは試す価値があるかも // TODO
- 事前学習済み VC モデルでパラレル音声を得るのではなく、 CycleGAN みたいにすればうまくいくかもしれないと思った

## 試したこと

```
wandb_id. script_name <- derived_from : description
------------------------------------------------------------

#11. 09               : pitch, energy を使わない mel to mel での変換性能を確認
                        -> あまり良くない (spksim 0.41 / step 10000)
#19. 09aa             : upsampling, downsampling を行う
                        -> #11 より良くない (spksim 0.16 / step 10000)
#20. 09ab             : FFNBlock を大量に重ねる
                        -> あまり変わらない (spksim 0.39, step 10000)
#21. 09ab <- #20      : kdim, vdim のボトルネックを外し、 hard attn を soft attn にする
                        -> spksim 向上, attn が発音に沿ってない, 恒等写像化か学習不足か (spksim 0.62, step 10000)
#22. 09aa <- #19, #21 : モデルをデカくして、 kdim, vdim のボトルネックも外す
                        -> 低速化, attn が少数の ref frame に集中, バグあるかも (spksim 0.28, step 10000)
#28. 09b  <- #21      : cycle loss, adv loss を入れる
                        -> 学習を途中で中断させた, 恒等写像化してるかも (spksim 0.50, step 10000)
#30. 09b  <- #28      : 変換中の音声もログを残す, cycle loss の倍率を 150 から 1 にする
                        -> 学習が全く進まない (spksim 0.04, step 6000)
#31. 09b  <- #30      : cycle loss の倍率を 10 にする
                        -> 1600 step あたりで #30 と異なる挙動をして cycle loss が下がった
                           generator が強度を反転するような変換を学習していた
                           わけわからない方法で情報を声風のフォーマットに埋め込んでる感じがする (spksim 0.04 step 6000)
#34. 09b  <- #30      : エネルギーが変換前後で一致するようにロスを加えた
                        -> cycle loss の倍率を 10 倍にし忘れたのもあって失敗 (spksim 0.12 step 4000)
#35. 09b  <- #34      : cycle loss の設定を 10 にして再度学習
                        -> ... (spksim 0.45 step 20000)
#38. 09ba <- #31      : 同一話者への変換が恒等写像になるようにロスを加える
                        -> ... (spksim 0.61 step 20000)

#47. 09ba <- #38      : speaker discriminator の出力を複数にした
                        -> spd_loss が下がらない, バグあるかも
                           実際 ACDiscriminator の normalize 軸を間違えてた
#48. 09b  <- #35      : speaker discriminator の出力を複数にした
                        -> 同上
#49. 09b  <- #48      : バグを修正した
                        -> spd_loss が下がらない, ノーマライズすべきじゃないかもしれない
#50. 09ba <- #47      : バグを修正した
                        -> 同上
#51. 09b  <- #49      : SPD の normalization をなくした
                        -> ...
#52. 09ba <- #50      : SPD の normalization をなくした
                        -> ...
#... 再試行

: D にて downsampling する
: mel-input, hubert-input の二種のエンコーダーを用意する
: InstanceNorm と比較
: RNN の効果を検証
: AdaIN の効果を検証
: hubert-soft でガイドするなど
: augmented cyclegan のアイデアを入れる
```

## 参考・関連

### [CycleGAN-VC2](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html), [arXiv](https://arxiv.org/abs/1904.04631) (2019)

Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo

- mel-cepstrum to mel-cepstrum
- Two-Step Adversarial Losses
  - over-smoothing を何とかするらしい
  - 話者をサイクルさせて元の話者に戻してから adv. loss を出す
- 2-1-2D CNN
  - 曰く `A 1D CNN is more feasible for capturing dynamical change, as it can capture the overall relationship along with the feature dimension. In contrast, a 2D CNN is better suited for converting features while preserving the original structures, as it restricts the converted region to local`
  - ablation study の結果を見た感じでは一番効果が大きそう
- PatchGAN
  - よくわかってない
  - 一部を切り出して D にかけるのではなくて、 D に短めのセグメントごとに出力させてる？

### [CycleGAN-VC3](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html), [arXiv](https://arxiv.org/abs/2010.11672) (2020)

Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo

- mel-spectrogram to mel-spectrogram
- Time-Frequency Adaptive Normalization
  - V1, V2 で time-freq structure が保持されないのを解決したらしい
    - ということは、何も工夫しないでやるとうまくいかないかも？
    - V2 までは mel-cepstrum でやってたらしいし
  - ソース melspec を使って AdaIN みたいなことをするらしい
  - 曰く ...
    - `TFAN normalizes it in a channel-wise manner similar to IN and then modulates the normalized feature in an element-wise manner using scale γ(x) and bias β(x)`
    - `In IN, x-independent scale β and bias γ are applied in a channel-wise manner, whereas in TFAN, those calculated from x (i.e., β(x) and γ(x)) are applied in an element-wise manner. These differences allow TFAN to adjust the scale and bias of f while reflecting x in a time and frequency-wise manner.`
- デモを聞く感じ、 V1, V2 間での改善が大きい気がする

### [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC), [arXiv](https://arxiv.org/abs/2107.10394) (2021)

Yinghao Aaron Li, Ali Zare, Nima Mesgarani
