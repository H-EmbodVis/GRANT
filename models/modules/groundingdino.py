# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from models.modules.detr_layers import DetrTransformerEncoderLayer

import math

def get_text_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. For example,
            input tensor is [x,y], the results will be [pos(y), pos(x)].
            Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') /
                          num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()),
                            dim=3).flatten(2)
        return sin_x

    pos_res = [
        sine_func(x)
        for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    ]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)


class BiFormer(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        num_biformers,
        dropout=0.1,
    ):
        super(BiFormer, self).__init__()

        self.self_attentions = nn.ModuleList()
        self.bi_attentions = nn.ModuleList()
        self.num_biformers = num_biformers
        self.word_self_attentions = nn.ModuleList()

        for i in range(num_biformers):
            bi_attention = BiAttentionBlock(v_dim, l_dim, embed_dim, num_heads, dropout=dropout)
            self.bi_attentions.append(bi_attention)
            self_attention = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=8, embed_dims=128, dropout=0.1, batch_first=True),
                ffn_cfg=dict(embed_dims=128, feedforward_channels=1024, ffn_drop=0.1),
            )
            self.self_attentions.append(self_attention)
            word_self_attention = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=8, embed_dims=128, dropout=0.1, batch_first=True),
                ffn_cfg=dict(embed_dims=128, feedforward_channels=1024, ffn_drop=0.1),
            )
            self.word_self_attentions.append(word_self_attention)
            
    def forward(self, v, l, v_pos=None, attention_mask_v=None, attention_mask_l=None):
        
        batch_size = v.shape[0]
        for i in range(self.num_biformers):
            num_concat_texts = l['embedded'].shape[0] // batch_size

            language_embedded = l['embedded']
            language_embedded = language_embedded.reshape(batch_size, -1, *language_embedded.shape[1:]).flatten(1,2)

            text_token_mask = l['text_token_mask']
            text_token_mask = text_token_mask.reshape(batch_size, -1, text_token_mask.shape[1])

            position_ids = l['position_ids']
            position_ids = position_ids.reshape(batch_size, -1, position_ids.shape[1])

            v, language_embedded = self.bi_attentions[i](v, language_embedded,
                                        attention_mask_v=None, attention_mask_l=~text_token_mask.flatten(1,2)) # 0 allow 
            
            language_embedded = language_embedded.reshape(batch_size * num_concat_texts, -1, language_embedded.shape[-1])
            position_ids = position_ids.reshape(batch_size * num_concat_texts, position_ids.shape[-1])
            pos_text = get_text_sine_pos_embed(
                position_ids[..., None],
                num_pos_feats=128,
                exchange_xy=False)

            language_embedded = self.word_self_attentions[i](
                query=language_embedded,
                query_pos=pos_text,
                attn_masks=(~l['masks']).repeat(self.word_self_attentions[i].self_attn.num_heads, 1, 1),
                key_padding_mask=None,
            )

            l['embedded'] = language_embedded
            v = self.self_attentions[i](query=v, 
                                        query_pos=v_pos.permute((1, 0, 2)),
                                        attn_masks=None,
                                        key_padding_mask=None)
        return v, l
