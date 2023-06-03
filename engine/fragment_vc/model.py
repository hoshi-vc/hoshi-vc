from typing import List, Optional, Tuple

import torch.functional as F
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (BatchNorm1d, Conv1d, Dropout, LayerNorm, Linear, Module, MultiheadAttention, ReLU, Sequential, Tanh, TransformerEncoder)

class FragmentVC(Module):
  def __init__(self, d_model=512):
    super().__init__()

    self.unet = UnetBlock(d_model)

    self.smoothers = TransformerEncoder(Smoother(d_model, 2, 1024), num_layers=3)

    self.mel_linear = Linear(d_model, 80)

    self.post_net = Sequential(
        Conv1d(80, 512, kernel_size=5, padding=2),
        BatchNorm1d(512),
        Tanh(),
        Dropout(0.5),
        Conv1d(512, 512, kernel_size=5, padding=2),
        BatchNorm1d(512),
        Tanh(),
        Dropout(0.5),
        Conv1d(512, 512, kernel_size=5, padding=2),
        BatchNorm1d(512),
        Tanh(),
        Dropout(0.5),
        Conv1d(512, 512, kernel_size=5, padding=2),
        BatchNorm1d(512),
        Tanh(),
        Dropout(0.5),
        Conv1d(512, 80, kernel_size=5, padding=2),
        BatchNorm1d(80),
        Dropout(0.5),
    )

  def forward(
      self,
      srcs: Tensor,
      refs: Tensor,
      src_masks: Optional[Tensor] = None,
      ref_masks: Optional[Tensor] = None,
  ) -> Tuple[Tensor, List[Optional[Tensor]]]:
    """
    Forward function.

    Args:
      srcs: (batch, src_len, 768)
      src_masks: (batch, src_len)
      refs: (batch, 80, ref_len)
      ref_masks: (batch, ref_len)
    """

    # out: (src_len, batch, d_model)
    out, attns = self.unet(srcs, refs, src_masks=src_masks, ref_masks=ref_masks)

    # out: (src_len, batch, d_model)
    out = self.smoothers(out, src_key_padding_mask=src_masks)

    # out: (src_len, batch, 80)
    out = self.mel_linear(out)

    # out: (batch, 80, src_len)
    out = out.transpose(1, 0).transpose(2, 1)
    refined = self.post_net(out)
    out = out + refined

    # out: (batch, 80, src_len)
    return out, attns

class UnetBlock(Module):
  def __init__(self, d_model: int):
    super(UnetBlock, self).__init__()

    self.conv1 = Conv1d(80, d_model, 3, padding=1, padding_mode="replicate")
    self.conv2 = Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")
    self.conv3 = Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")

    self.prenet = Sequential(
        Linear(768, 768),
        ReLU(),
        Linear(768, d_model),
    )

    self.extractor1 = Extractor(d_model, 2, 1024, no_residual=True)
    self.extractor2 = Extractor(d_model, 2, 1024)
    self.extractor3 = Extractor(d_model, 2, 1024)

  def forward(
      self,
      srcs: Tensor,
      refs: Tensor,
      src_masks: Optional[Tensor] = None,
      ref_masks: Optional[Tensor] = None,
  ) -> Tuple[Tensor, List[Optional[Tensor]]]:
    """
    Args:
      srcs: (batch, src_len, 768)
      src_masks: (batch, src_len)
      refs: (batch, 80, ref_len)
      ref_masks: (batch, ref_len)
    """

    # tgt: (batch, tgt_len, d_model)
    tgt = self.prenet(srcs)
    # tgt: (tgt_len, batch, d_model)
    tgt = tgt.transpose(0, 1)

    # ref*: (batch, d_model, mel_len)
    ref1 = self.conv1(refs)
    ref2 = self.conv2(F.relu(ref1))
    ref3 = self.conv3(F.relu(ref2))

    # out*: (tgt_len, batch, d_model)
    out, attn1 = self.extractor1(
        tgt,
        ref3.transpose(1, 2).transpose(0, 1),
        tgt_key_padding_mask=src_masks,
        memory_key_padding_mask=ref_masks,
    )
    out, attn2 = self.extractor2(
        out,
        ref2.transpose(1, 2).transpose(0, 1),
        tgt_key_padding_mask=src_masks,
        memory_key_padding_mask=ref_masks,
    )
    out, attn3 = self.extractor3(
        out,
        ref1.transpose(1, 2).transpose(0, 1),
        tgt_key_padding_mask=src_masks,
        memory_key_padding_mask=ref_masks,
    )

    # out: (tgt_len, batch, d_model)
    return out, [attn1, attn2, attn3]

class Smoother(Module):
  def __init__(self, d_model: int, nhead: int, d_hid: int, dropout=0.1):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

    self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
    self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)

    self.norm1 = LayerNorm(d_model)
    self.norm2 = LayerNorm(d_model)
    self.dropout1 = Dropout(dropout)
    self.dropout2 = Dropout(dropout)

  def forward(
      self,
      src: Tensor,
      src_mask: Optional[Tensor] = None,
      is_causal: Optional[bool] = None,
      src_key_padding_mask: Optional[Tensor] = None,
  ) -> Tensor:
    assert src_mask is None

    # multi-head self attention
    src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

    # add & norm
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    # conv1d
    src2 = src.transpose(0, 1).transpose(1, 2)
    src2 = self.conv2(F.relu(self.conv1(src2)))
    src2 = src2.transpose(1, 2).transpose(0, 1)

    # add & norm
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src

class Extractor(Module):
  def __init__(
      self,
      d_model: int,
      nhead: int,
      d_hid: int,
      dropout=0.1,
      no_residual=False,
  ):
    super().__init__()

    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
    self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

    self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
    self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)

    self.norm1 = LayerNorm(d_model)
    self.norm2 = LayerNorm(d_model)
    self.norm3 = LayerNorm(d_model)
    self.dropout1 = Dropout(dropout)
    self.dropout2 = Dropout(dropout)
    self.dropout3 = Dropout(dropout)

    self.no_residual = no_residual

  def forward(
      self,
      tgt: Tensor,
      memory: Tensor,
      tgt_mask: Optional[Tensor] = None,
      memory_mask: Optional[Tensor] = None,
      tgt_key_padding_mask: Optional[Tensor] = None,
      memory_key_padding_mask: Optional[Tensor] = None,
  ) -> Tuple[Tensor, Optional[Tensor]]:
    # multi-head self attention
    tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

    # add & norm
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    # multi-head cross attention
    tgt2, attn = self.cross_attn(
        tgt,
        memory,
        memory,
        attn_mask=memory_mask,
        key_padding_mask=memory_key_padding_mask,
    )

    # add & norm
    if self.no_residual:
      tgt = self.dropout2(tgt2)
    else:
      tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)

    # conv1d
    tgt2 = tgt.transpose(0, 1).transpose(1, 2)
    tgt2 = self.conv2(F.relu(self.conv1(tgt2)))
    tgt2 = tgt2.transpose(1, 2).transpose(0, 1)

    # add & norm
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)

    return tgt, attn
