# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pyworld as pw

from engine.lib.utils import NPArray

def pyworld_vc(audio: NPArray, sr: int, f0_shift: float, sp_shift: float):

  # CREPE と同じで 20 cent 単位でシフトする
  # https://arxiv.org/pdf/1802.06182.pdf
  f0_shift_cent = 20 * f0_shift
  sp_shift_cent = 20 * sp_shift
  f0_scale = 2**(f0_shift_cent / 1200)
  sp_scale = 2**(sp_shift_cent / 1200)

  # sp: スペクトル包絡, ap: 非周期性指標
  audio = audio.astype(np.double)
  f0, t = pw.dio(audio, sr)
  f0 = pw.stonemask(audio, f0, t, sr)
  sp = pw.cheaptrick(audio, f0, t, sr)
  ap = pw.d4c(audio, f0, t, sr)

  next_f0 = f0_scale * f0

  next_sp = np.zeros_like(sp)
  sp_range = int(next_sp.shape[1] * sp_scale)
  for f in range(next_sp.shape[1]):
    if f < sp_range:
      next_sp[:, f] = sp[:, int(f / sp_scale)]
    else:
      next_sp[:, f] = sp[:, f]

  return pw.synthesize(next_f0, next_sp, ap, sr)
