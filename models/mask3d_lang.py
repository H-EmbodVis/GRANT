import torch
import hydra
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max
from torch.cuda.amp import autocast
from models.modules.groundingdino import BiFormer

class Mask3DLang(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_normal,
        bert_path="./bert-base-uncased",
        temperature=0.1,
        num_biformers=3,
        inter_biformers=0,
        aux_loss=True,
        language_model=False,
        lang_max_token_length=256,
        softmax_mode=False,
        axis_align_coord=False,
        **kwargs
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.aux_loss = aux_loss
        self.num_biformers = num_biformers
        
        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(
                mask, p2s, dim=dim
            )[0]
        else:
            assert False, "Scatter function not known"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temperature = temperature
        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )
        else:
            assert False, "pos enc type not known"

        self.pooling = MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3
        )

        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()
        self.each_decoder_norm = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            meta_class_former = nn.ModuleList()
            lang_former = nn.ModuleList()
            tmp_norm = nn.ModuleList()
            bi_former = nn.ModuleList()

            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(sizes[hlevel], self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )
                tmp_norm.append(nn.LayerNorm(hidden_dim))
            
            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)
            self.each_decoder_norm.append(tmp_norm)
            
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.decoder_norm_extra = nn.LayerNorm(hidden_dim)
        self.extra_query_instance_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_instance_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        language_model=dict(
            name=bert_path,
            pad_to_max=False,
            use_sub_sentence_represent=True,
            special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
            add_pooling_layer=True,
            max_tokens=lang_max_token_length,
        )
        
        from models.modules.bert import BertModel
        self.language_model = BertModel(**language_model)

        self.text_feat_map = nn.Linear(
            768,
            hidden_dim,
            bias=True)

        if self.num_biformers > 0:
            self.bi_former = nn.ModuleList()
            for _ in range(num_shared):
                bi_former = BiFormer(self.mask_dim, self.mask_dim, dim_feedforward, self.num_heads, self.num_biformers)
                self.bi_former.append(bi_former)
        if inter_biformers > 0:
            self.inter_biformers = nn.ModuleList()
            for _ in range(num_shared):
                self.inter_biformers.append(nn.ModuleList())
                for i, hlevel in enumerate(self.hlevels):
                    self.inter_biformers[-1].append(
                        BiFormer(self.mask_dim, self.mask_dim, dim_feedforward, self.num_heads, inter_biformers)
                    )
        else:
            self.inter_biformers = None

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(
        self, x, point2segment=None, raw_coordinates=None, extra_lang=None, is_eval=False
    ):
        pcd_features, aux = self.backbone(x)

        batch_size = len(x.decomposed_coordinates)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)

        if self.train_on_segments:
            mask_segments = []
            for i, mask_feature in enumerate(
                mask_features.decomposed_features
            ):
                mask_segments.append(
                    self.scatter_fn(mask_feature, point2segment[i], dim=0)
                )

        sampled_coords = None

        # ================ FPS sampling =======================
        assert self.non_parametric_queries
        fps_idx = [
            furthest_point_sample(
                x.decomposed_coordinates[i][None, ...].float(),
                self.num_queries,
            )
            .squeeze(0)
            .long()
            for i in range(len(x.decomposed_coordinates))
        ]
            
        sampled_coords = torch.stack(
            [
                coordinates.decomposed_features[i][fps_idx[i].long(), :]
                for i in range(len(fps_idx))
            ]
        ).float()

        mins = torch.stack(
            [
                coordinates.decomposed_features[i].min(dim=0)[0]
                for i in range(len(coordinates.decomposed_features))
            ]
        )
        maxs = torch.stack(
            [
                coordinates.decomposed_features[i].max(dim=0)[0]
                for i in range(len(coordinates.decomposed_features))
            ]
        )

        query_pos = self.pos_enc(
            sampled_coords, input_range=[mins, maxs]
        )  
        query_pos = self.query_projection(query_pos)

        queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))


        extra_queries = None
        if extra_lang is not None:
            batch_num_concat_texts = torch.as_tensor(extra_lang.batch_num_concat_texts).cuda()
            extra_queries = self.language_model(extra_lang.batch_concat_texts) 
            extra_queries['embedded'] = self.text_feat_map(extra_queries['embedded'])
            extra_queries['extra_lang'] = extra_lang

        predictions_class = []
        predictions_mask = []
        all_normalized_queries_embed = []

        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                if self.train_on_segments:
                    output_class, outputs_mask, attn_mask, normalized_queries_embed = self.mask_module(
                        queries,
                        mask_features,
                        mask_segments,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=point2segment,
                        extra_queries=extra_queries,
                        return_normalized_query_embed=True,
                        query_coords=sampled_coords,
                    )
                else:
                    output_class, outputs_mask, attn_mask, normalized_queries_embed = self.mask_module(
                        queries,
                        mask_features,
                        None,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=None,
                        extra_queries=extra_queries,
                        return_normalized_query_embed=True,
                        query_coords=sampled_coords,
                    )

                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)
                all_normalized_queries_embed.append(normalized_queries_embed)
                
                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(
                        curr_sample_size, self.sample_sizes[hlevel]
                    )

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(
                            pcd_size, device=queries.device
                        )

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        idx = torch.randperm(
                            decomposed_aux[k].shape[0], device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [
                        decomposed_aux[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn = torch.stack(
                    [
                        decomposed_attn[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](
                    batched_aux.permute((1, 0, 2))
                )
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                queries = self.each_decoder_norm[decoder_counter][i](queries)

                if self.inter_biformers is not None:
                    queries, extra_queries = self.inter_biformers[decoder_counter][i](queries, extra_queries, v_pos=query_pos)

            if self.num_biformers > 0:
                queries, extra_queries = self.bi_former[decoder_counter](queries, extra_queries, v_pos=query_pos)

        if self.train_on_segments:
            output_class, outputs_mask, normalized_queries_embed = self.mask_module(
                queries,
                mask_features,
                mask_segments,
                0,
                ret_attn_mask=False,
                point2segment=point2segment,
                extra_queries=extra_queries,
                return_normalized_query_embed=True,
                query_coords=sampled_coords,
            )
        else:
            output_class, outputs_mask, normalized_queries_embed = self.mask_module(
                queries,
                mask_features,
                None,
                0,
                ret_attn_mask=False,
                point2segment=None,
                extra_queries=extra_queries,
                return_normalized_query_embed=True,
                query_coords=sampled_coords,
            )
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        all_normalized_queries_embed.append(normalized_queries_embed)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask
            ) if self.aux_loss else [],
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
            "queries_hidden_state": queries,
            "queries_normalized_embed": normalized_queries_embed,
            "extra_queries": extra_queries,
        }

    def mask_module(
        self,
        query_feat,
        mask_features,
        mask_segments,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        extra_queries=None,
        return_normalized_query_embed=False,
        query_coords=None,
    ):
        batch_size, num_queries = query_feat.shape[:2]
        query_feat = self.decoder_norm(query_feat)

        mask_embed = self.mask_embed_head(query_feat)

        query_embed = self.query_instance_embed_head(query_feat)
        normalized_query_embed = query_embed / (query_embed.norm(dim=-1, keepdim=True) + 1e-8) 
        
        if extra_queries: 
            batch_concat_positive_map, batch_token_inst_ids_pair = extra_queries['extra_lang'].concat_lang_positive_maps, extra_queries['extra_lang'].unflatten_lang_token_inst_id_pairs

            batch_text_token_feat = extra_queries['embedded'] 
            batch_text_token_feat = self.decoder_norm_extra(batch_text_token_feat)
            batch_text_token_feat = self.extra_query_instance_embed_head(batch_text_token_feat) 

            outputs_class_batch = [] 
            for bid, (positive_map, _) in enumerate(zip(batch_concat_positive_map, batch_token_inst_ids_pair)):
                num_concat_texts = positive_map.shape[0]
                text_token_feat = batch_text_token_feat[num_concat_texts * bid: num_concat_texts * bid + num_concat_texts]

                positive_map = positive_map.flatten(0, 1).float().T.cuda()
                positive_token_embed = (positive_map @ text_token_feat.flatten(0, 1)) / (positive_map.sum(dim=1, keepdim=True) + 1e-8)
                normalized_positive_token_embed = positive_token_embed / (positive_token_embed.norm(dim=-1, keepdim=True) + 1e-8) 

                output_class = (normalized_query_embed[bid] @ normalized_positive_token_embed.T) / self.temperature 

                invalid_text_token = (positive_map.sum(dim=1) < 1e-8)
                output_class[:, invalid_text_token] = -torch.inf

                outputs_class_batch.append(output_class)

            outputs_class = outputs_class_batch

        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(
                    mask_features.decomposed_features[i] @ mask_embed[i].T
                )

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        ret = [
            (outputs_class if isinstance(outputs_class, torch.Tensor) else [i for i in outputs_class]) if extra_queries else None,
            [i for i in output_segments] if point2segment is not None else [i for i in outputs_mask.decomposed_features],
        ]

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            ) 

            ret.append(attn_mask)

        if return_normalized_query_embed:
            ret.append(normalized_query_embed)

        return ret

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
