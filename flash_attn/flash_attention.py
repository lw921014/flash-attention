import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from flash_attn.rotary import RotaryEmbedding, RotaryEmbedding2D
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

import numpy as np

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False, pos_bias = None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        assert not need_weights
        assert qkv.dtype == torch.float16
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, pos_bias, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, attn_mask=attn_mask
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                key_padding_mask_bool = key_padding_mask.bool_matrix
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask_bool)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, pos_bias,cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, attn_mask=attn_mask
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, pos_bias, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal, attn_mask=attn_mask
            )

        return output


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, window_size, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, use_rotary_emb=None, device=None, dtype=None, init_para = {}, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.window_size = window_size  # Wh, Ww

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        assert use_rotary_emb in [None, '1d', '2d']
        self.use_rotary_emb = use_rotary_emb
        if self.use_rotary_emb == '1d':
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        elif self.use_rotary_emb == '2d':
            self.rotary_emb = RotaryEmbedding2D(self.head_dim)

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if 'qkv_weight' in init_para.keys():
            self.Wqkv.weight = torch.nn.Parameter(init_para["qkv_weight"])
        if 'qkv_bias' in init_para.keys():
            self.Wqkv.bias = torch.nn.Parameter(init_para["qkv_bias"])
        if 'proj_weight' in init_para.keys():
            self.out_proj.weight = torch.nn.Parameter(init_para["proj_weight"])
        if 'proj_bias' in init_para.keys():
            self.out_proj.bias = torch.nn.Parameter(init_para["proj_bias"])

        # for pos bias
            # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # TODO(liuwei88): may need this in the future
        # runc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, x_ignored_, x_ignored_1_, attn_mask=None, key_padding_mask=None,
                need_weights=False, pos_bias=None):
        qkv = self.Wqkv(x)
        if self.use_rotary_emb:
            query, key, value = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3,
                                          h=self.num_heads).unbind(dim=2)
            query, key = self.rotary_emb(query, key, seq_dimension=-3)
            qkv = torch.stack([query, key, value], dim=2)
        else:
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # NOTE: add some hard code for quick test in Swin-T, need to fix it in the future
        # forcely padding mask seq_q and seq_k to 64, keep other dim the same
        relative_position_bias = F.pad(relative_position_bias, (0, 15, 0, 15), mode='constant', value=0).contiguous()

        # NOTE: add some hard code for quick test in Swin-T, need to fix it in the future
        # forcely padding mask seq_q and seq_k to 64, keep other dim the same
        if attn_mask is not None:
            # print(attn_mask.size())
            attn_mask = F.pad(attn_mask, (0, 15, 0, 15), mode='constant', value=0).contiguous()

        context = self.inner_attn(qkv, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal,
                                                pos_bias = relative_position_bias)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))
