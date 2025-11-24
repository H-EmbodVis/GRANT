
import gc
import statistics
import shutil
import os
import os.path as osp
import math
from loguru import logger
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from datetime import datetime

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from hashlib import md5
import glob
import pickle
import json

from datasets.language_info import lang_info_data

from datasets.scannet200.scannet200_splits import (
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)
from benchmark.evaluate_semantic_instance import evaluate

from models.metrics import IoU
from models.metrics.evaluate_LLM import eval_llm_iou_score
from models.misc import get_batch_aabb_pair_ious, logical_or_sum, get_evenly_distributed_colors, fix_seed, print_grad_status
from utils.votenet_utils.eval_det import eval_det

from models.metrics.utils import eval_seg_model, collect_grounding_score
from transformers import AutoConfig


class ModelingGrounded3DLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments
        self.config = config


        self.tmpdir = osp.join(
            './.dist_test', md5(self.config.general.save_dir.encode()).hexdigest())
        os.makedirs(self.tmpdir, exist_ok=True)
        for i in glob.glob(self.tmpdir + '/*'):
            os.remove(i)

        self.save_hyperparameters()


        self.init_llama_model() 


        self.model = hydra.utils.instantiate(config.model) 

        if self.llama_config.llm_only: 

            for name, param in self.model.named_parameters(): 
                param.requires_grad = False


        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(
            config.matcher,
        )
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict,
        )

        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()

        self.labels_info = dict()

        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset,
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset,
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset,
        )

        self.labels_info = self.train_dataset.label_info

        self.automatic_optimization = False 

    def init_llama_model(self):
        save_folder_name = datetime.now().strftime(
            "%m-%d-%H-%M-%S") if not self.config.general.timestamp else self.config.general.timestamp

        llama_config = AutoConfig.from_pretrained(
            self.config.general.llm_config)
        llama_config.save_path = f"{self.config['general']['save_dir']}/{save_folder_name}"

        if llama_config.enable_llm:
            print('*****************************************************************')
            print(f'Using config: {self.config.general.llm_config}')
            print(f'Using data config: {self.config.general.llm_data_config}')
            self.data_to_load = json.load(
                open(self.config.general.llm_data_config))
            print('*****************************************************************')
            
            if self.global_rank == 0:
                llama_config.save_pretrained(
                    f"{self.config['general']['save_dir']}")
            if not llama_config.load_pretrain_weight:
                logger.warning(
                    f"llm pretrain weight is not loaded: do you need to debug or resume from last_epoch.ckpt !?"
                )

            os.makedirs(llama_config.save_path, exist_ok=True)
            os.makedirs(f"{llama_config.save_path}/m3drefer", exist_ok=True)

            assert not self.config.general.use_dbscan
            # init tokenizer and add special tokens
            from models.LLM.LLama3d import load_llama_model_and_tokenizer
            self.llama_model, self.llama_tokenizer = load_llama_model_and_tokenizer(
                llama_config)
        else: 
            print(" ====================== llm is disabled ===================")

        self.llama_config = llama_config

    def forward(
        self, x, point2segment=None, raw_coordinates=None, extra_lang=None, is_eval=False
    ):
        x = self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            extra_lang=extra_lang
        )
        return x

    def prepare_llm(self, output, extra_lang, assigner_indices=None, target=None, raw_data=None, file_names=None):
        batch_lang_infos = []
        batch_map_target_to_query = []

        batch_size = output['queries_hidden_state'].shape[0]

        total_concat_texts = 0
        for bid in range(batch_size):
            raw_texts_bid = ''.join(extra_lang.batch_concat_texts[total_concat_texts:total_concat_texts +
                                    extra_lang.batch_num_concat_texts[bid]]).split('. ')[:-1]  
            if output['extra_queries'] is not None:
                assert extra_lang.batch_num_concat_texts[bid] == output[
                    'extra_queries']['embedded'].shape[0] // batch_size

                # get the start and end token id for each sentence
                each_lang_query_features = []
                for concat_text_id, (concat_text_pos_ids, text_token_mask) in enumerate(zip(
                    output['extra_queries']['position_ids'][total_concat_texts:
                                                            total_concat_texts+extra_lang.batch_num_concat_texts[bid]],
                    output['extra_queries']['text_token_mask'][total_concat_texts:
                                                               total_concat_texts+extra_lang.batch_num_concat_texts[bid]]
                )):
                    each_concat_text_features = []
                    start_i_of_pos_ids = 0
                    while start_i_of_pos_ids < len(concat_text_pos_ids) and text_token_mask[start_i_of_pos_ids]:
                        i = start_i_of_pos_ids + 1
                        while i < len(concat_text_pos_ids) and text_token_mask[i] and concat_text_pos_ids[i-1] < concat_text_pos_ids[i]:
                            i += 1
                        each_concat_text_features.append(
                            output['extra_queries']['embedded'][total_concat_texts + concat_text_id, start_i_of_pos_ids:i])
                        start_i_of_pos_ids = i
                    each_lang_query_features.extend(
                        each_concat_text_features[1:-1])  # drop first 0 and last 0
                # get targets for each sentence
                flatten_lang_token_inst_id_pair = extra_lang.flatten_lang_token_inst_id_pairs[
                    bid]
                raw_lang_type = extra_lang.raw_lang_types[bid]
                assert len(flatten_lang_token_inst_id_pair) == len(raw_lang_type) == len(
                    each_lang_query_features) == len(raw_texts_bid), 'length does not match'
            else:
                # get targets for each sentence
                flatten_lang_token_inst_id_pair = extra_lang.flatten_lang_token_inst_id_pairs[
                    bid]
                raw_lang_type = extra_lang.raw_lang_types[bid]
                each_lang_query_features = [None] * \
                    len(extra_lang.raw_lang_types[bid])
                raw_texts_bid = raw_texts_bid + \
                    [None] * (len(extra_lang.raw_lang_types[bid]) -
                              len(raw_texts_bid))

            # ---------- compute target to query mapping -------------
            assert self.llama_config.valid_target_iou >= 0., 'The matched iou should be larger than 0.'
            if self.model.train_on_segments:
                pred_masks = (
                    output["pred_masks"][bid]
                    .detach()
                    .cpu()[target[bid]["point2segment"].cpu()]
                )  # map back to raw points
            else:
                pred_masks = (
                    output["pred_masks"][bid]
                    .detach()
                    .cpu()
                )

            if not self.training:  # evaluation use box iou
                target_mask = target[bid]['masks'].cpu().float()

                # box iou
                target_boxes = []
                full_res_target_mask = target_mask[:,
                                                   raw_data.inverse_maps[0]]
                for mask in full_res_target_mask:
                    gt_points = raw_data.full_res_coords[0][mask > 0.5]
                    min_vals, max_vals = gt_points.min(
                        axis=0), gt_points.max(axis=0)
                    target_boxes.append([min_vals, max_vals])
                target_boxes = torch.as_tensor(target_boxes)

                pred_boxes = []
                full_res_pred_mask = pred_masks.T[:,
                                                  raw_data.inverse_maps[0]]
                for mask in full_res_pred_mask:
                    gt_points = raw_data.full_res_coords[0][mask > 0.5]
                    if gt_points.shape[0] > 0:
                        min_vals, max_vals = gt_points.min(
                            axis=0), gt_points.max(axis=0)
                    else:
                        min_vals, max_vals = np.zeros((3,)), np.zeros((3,))
                    pred_boxes.append([min_vals, max_vals])
                pred_boxes = torch.as_tensor(pred_boxes)

                box_iou = torch.zeros(
                    (len(target_boxes), len(pred_boxes)), device='cpu')
                for i, tb in enumerate(target_boxes):
                    for j, pb in enumerate(pred_boxes):
                        box_iou[i, j] = get_batch_aabb_pair_ious(
                            tb[None], pb[None])

                map_target_to_query = box_iou.argmax(1)
                gt_ious = box_iou.max(1)[0]

                # following LL3DA, Scan2Cap, Vote2Cap-DETR++
                max_query_iou, max_query_iou_gt_id = box_iou.max(
                    dim=0)  # for nqueries
                tmpiou = torch.zeros_like(box_iou)
                tmpiou[max_query_iou_gt_id, torch.arange(
                    tmpiou.shape[1])] = max_query_iou
                max_gt_iou, max_gt_iou_query_id = tmpiou.max(
                    dim=1)  # find the maximum gt

                valid_target = np.ones_like(
                    map_target_to_query, dtype=bool)
            else:
                inter = (target[bid]['masks'].cpu().float()
                         @ (pred_masks > 0).float())
                outer = logical_or_sum(
                    target[bid]['masks'].cpu(), (pred_masks.T > 0))
                iou = inter / (outer + 1e-8)
                map_target_to_query = iou.argmax(1)
                gt_ious = iou.max(1)[0]
                valid_target = gt_ious >= self.llama_config.valid_target_iou 

            batch_map_target_to_query.append(
                [map_target_to_query, valid_target])
            
            if self.training:
                from utils.sample_utils import sample_by_type
                max_sample_lang_type_count = dict(detection=self.data_to_load["detection"],
                                                  scanrefer=self.data_to_load["scanrefer"],
                                                  m3dref=self.data_to_load["m3dref"],
                                                  groundedscenecaption=self.data_to_load.get(
                                                      "groundedscenecaption", 0),
                                                    taskscheduling=self.data_to_load.get(
                                                      "taskscheduling", 0)
                                                      )
                lang_type_with_index = np.asarray(
                    [(d.split(':')[0], i) for i, d in enumerate(raw_lang_type)], dtype=object)
                sampled_lang_type_with_index = sample_by_type(
                    lang_type_with_index, max_sample_lang_type_count)
                sampled_indices = sampled_lang_type_with_index[:, 1]

            elif self.config.general.eval_type_range == "one_sample_each_type":
                unique_strings = set()
                sampled_indices = []
                # Iterate through the list and select indices for first occurrences
                for i, item in enumerate(raw_lang_type):
                    first_part = item.split(':')[0]  # Get the string before the first colon
                    if first_part not in unique_strings:
                        unique_strings.add(first_part)  # Add it to the set of seen strings
                        sampled_indices.append(i)  # Append the index of the first occurrence       
            elif self.config.general.eval_type_range == "most_1_dectection":
                unique_strings = set()
                sampled_indices = []
                least_one = False
                # Iterate through the list and select indices for first occurrences
                for i, item in enumerate(raw_lang_type):
                    first_part = item.split(':')[0]  # Get the string before the first colon
                    if first_part not in unique_strings or not least_one:
                        unique_strings.add(first_part)  # Add it to the set of seen strings
                        sampled_indices.append(i)  # Append the index of the first occurrence  
                        least_one = True
            elif self.config.general.eval_type_range == "only_taskscheduling":
                sampled_indices = []
                # Iterate through the list and select indices for first occurrences
                for i, item in enumerate(raw_lang_type):
                    first_part = item.split(':')[0]  # Get the string before the first colon
                    if first_part == "taskscheduling":
                        sampled_indices.append(i)  # Append the index of the first occurrence  

            else:
                sampled_indices = range(len(raw_lang_type)) 

            for sample_idx in sampled_indices:
                lang_token_inst_id_pair, lang_text, lang_type, lang_feat = flatten_lang_token_inst_id_pair[
                    sample_idx], raw_texts_bid[sample_idx], raw_lang_type[sample_idx], each_lang_query_features[sample_idx]
                lang_info = lang_info_data.from_grounding(
                    raw_text=lang_text,
                    lang_type=lang_type,
                    lang_token_inst_id_pair=lang_token_inst_id_pair,
                    map_target_to_query=map_target_to_query,
                    valid_target=valid_target,
                    support_counting=getattr(
                        self.llama_config, "support_counting", False),
                    count_instance=getattr(
                        self.llama_config, "count_instance", True),
                )
                lang_info.append_prompt_postfix() 
                lang_info.set_context_features(
                    query_hidden_feature=output['queries_hidden_state'][bid],
                    query_normalized_embed=output['queries_normalized_embed'][bid],
                )
                lang_info.set_batch_idx(bid)
                batch_lang_infos.append(lang_info)

            if 'scanqa' in self.config.data.lang_data_conf or 'objdesc' in self.config.data.lang_data_conf or 'scenedesc' in self.config.data.lang_data_conf or \
                    'scan2cap' in self.config.data.lang_data_conf or '3dllm' in self.config.data.lang_data_conf or 'embodiedplan' in self.config.data.lang_data_conf or\
                    'embodieddialog' in self.config.data.lang_data_conf or 'taskscheduling' in self.config.data.lang_data_conf:
                for i, lang_info in enumerate(raw_data.extra_qa[bid]):
                    lang_info.set_context_features(
                        query_hidden_feature=output['queries_hidden_state'][bid],
                        query_normalized_embed=output['queries_normalized_embed'][bid],
                    )

                    try:
                        if ('scan2cap' in lang_info.lang_type or 'objdesc' in lang_info.lang_type) and not self.training:
                            mapping = max_gt_iou_query_id
                        else:
                            mapping = map_target_to_query

                        lang_info.query_ids_question = []
                        lang_info.query_ids_answer = []
                        for inst_ids in lang_info.inst_ids_question:
                            lang_info.query_ids_question.append(
                                mapping[inst_ids][valid_target[inst_ids]].tolist()) 
                        # BUG    
                        for inst_ids in lang_info.inst_ids_answer:
                            lang_info.query_ids_answer.append(
                                mapping[inst_ids][valid_target[inst_ids]].tolist())

                    except Exception as e:
                        print(f'+++++ {lang_info.lang_type}: {e}')
                        from IPython import embed
                        embed()

                    lang_info.append_prompt_postfix()
                    lang_info.set_batch_idx(bid)
                    if not self.training:
                        lang_info.set_max_gt_iou(max_gt_iou)
                    batch_lang_infos.append(lang_info)

            total_concat_texts += extra_lang.batch_num_concat_texts[bid]


        return batch_lang_infos, batch_map_target_to_query

    def training_step(self, batch, batch_idx):
        raw_data, target, file_names = batch

        optimizer = self.optimizers()
        optimizer.zero_grad()

        if len(target) == 0:
            print("no targets")
            return None 

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = raw_data.features[:, -3:]
            raw_data.features = raw_data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=raw_data.coordinates,
            features=raw_data.features,
            device=self.device,
        )

        output = self.forward(
            data,
            point2segment=[
                target[i]["point2segment"] for i in range(len(target))
            ],
            raw_coordinates=raw_coordinates,
            extra_lang=raw_data.extra_lang 
        )

        output['raw_coordinates'] = raw_coordinates
        del raw_coordinates
        torch.cuda.empty_cache()

        if len(raw_data.extra_lang) > 0:
            output['extra_lang'] = raw_data.extra_lang
            if 'aux_outputs' in output:
                for aux_outputs in output["aux_outputs"]:
                    aux_outputs['extra_lang'] = raw_data.extra_lang
        losses = self.criterion(output, target, mask_type=self.mask_type) 


        if self.llama_config.enable_llm: 
            batch_lang_infos, batch_map_target_to_query = \
                self.prepare_llm(output, raw_data.extra_lang,
                                 None, target, raw_data)

            batch_gt_inst_ids = [((i.batch_idx, i, i.max_gt_iou) if not self.training else (
                i.batch_idx, i)) for i in batch_lang_infos]
            batch_input_texts = [i.question for i in batch_lang_infos]
            batch_output_texts = [i.answer for i in batch_lang_infos] 
            batch_eval_types = [i.lang_type for i in batch_lang_infos]
            batch_instance_queries_hidden_state = [
                i.query_hidden_feature for i in batch_lang_infos]
            batch_instance_queries_normalized_embed = [
                i.query_normalized_embed for i in batch_lang_infos]

            batch_output_think_texts = [i.thinking for i in batch_lang_infos]
            batch_min_time = [i.min_time for i in batch_lang_infos]
            batch_min_time_action_list = [i.min_time_action_list for i in batch_lang_infos]
            output_llm = self.llama_model(batch_input_text_list=batch_input_texts,
                                        batch_output_text_list=batch_output_texts,
                                        batch_instance_queries_hidden_state=batch_instance_queries_hidden_state,
                                        batch_instance_queries_normalized_embed=batch_instance_queries_normalized_embed,
                                        batch_eval_types=batch_eval_types,
                                        batch_gt_inst_ids=batch_gt_inst_ids,
                                        batch_output_think_texts=batch_output_think_texts,
                                        batch_min_time=batch_min_time,
                                        batch_min_time_action_list=batch_min_time_action_list,
                                        )
            if output_llm == None:
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.manual_backward(total_loss)
                self.clip_gradients(optimizer, gradient_clip_val=0.1,
                                    gradient_clip_algorithm="norm")
                
                optimizer.step()

                lr_scheduler = self.lr_schedulers()
                if self.config.scheduler.pytorch_lightning_params.interval == 'step':
                    lr_scheduler.step()
                elif self.config.scheduler.pytorch_lightning_params.interval == 'epoch':
                    if self.trainer.is_last_batch:
                        lr_scheduler.step()
                else:
                    raise NotImplementedError
                return total_loss.detach()  
            
            output['output_llm'] = output_llm

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        if self.llama_config.enable_llm:
            for k, v in output_llm.items():
                if 'loss' in k:
                    losses[k] = v

        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        if self.llama_config.enable_llm:
            logs["train_mean_loss_lm"] = output_llm["lm_loss"] 
            logs["train_mean_loss_lm_match"] = output_llm["match_loss"] 
            if self.llama_config.use_reason_loss:
                logs["train_mean_loss_lm_reason"] = output_llm["reason_loss"]
                self.log("reason_loss", output_llm["reason_loss"],
                     on_step=True, on_epoch=True, logger=True, batch_size=len(batch_eval_types))
            self.log("lm_loss", output_llm["lm_loss"],
                     on_step=True, on_epoch=True,  logger=True, batch_size=len(batch_eval_types))
            self.log("match_loss", output_llm["match_loss"],
                     on_step=True, on_epoch=True, logger=True, batch_size=len(batch_eval_types))

        self.log_dict(logs)
        total_loss = sum(losses.values())
        self.manual_backward(total_loss)
        del data, output, losses, output_llm
        torch.cuda.empty_cache()
        self.clip_gradients(optimizer, gradient_clip_val=0.1,
                            gradient_clip_algorithm="norm")

        optimizer.step()

        lr_scheduler = self.lr_schedulers()
        if self.config.scheduler.pytorch_lightning_params.interval == 'step':
            lr_scheduler.step()
        elif self.config.scheduler.pytorch_lightning_params.interval == 'epoch':
            if self.trainer.is_last_batch:
                lr_scheduler.step()
        else:
            raise NotImplementedError
        return total_loss.detach()

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        base_path = f"saved/eval_output/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        from pathlib import Path
        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > 1e-4:
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def save_visualizations(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        query_text=None, 
        query_mask=None, 
        query_mask_instance_coordscore=None,
        gt_query_mask=None, # 
        gt_ious=None,
        max_num_of_queries=200,
        max_num_of_instances=40,
    ):

        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []
        if "labels" in target_full: 
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask) in enumerate(
                zip(target_full["labels"], target_full["masks"])
            ):
                if label == 255:
                    continue

                mask_tmp = mask.detach().cpu().numpy()
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue

                gt_pcd_pos.append(mask_coords)
                mask_coords_min = full_res_coords[
                    mask_tmp.astype(bool), :
                ].min(axis=0)
                mask_coords_max = full_res_coords[
                    mask_tmp.astype(bool), :
                ].max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append(
                    {
                        "position": mask_coords_middle,
                        "size": size,
                        "color": self.validation_dataset.map2color([label])[0],
                    }
                )

                gt_pcd_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        gt_pcd_pos[-1].shape[0], 1
                    )
                )
                gt_inst_pcd_color.append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(gt_pcd_pos[-1].shape[0], 1)
                )

                gt_pcd_normals.append(
                    original_normals[mask_tmp.astype(bool), :]
                )

            gt_pcd_pos = np.concatenate(gt_pcd_pos)
            gt_pcd_normals = np.concatenate(gt_pcd_normals)
            gt_pcd_color = np.concatenate(gt_pcd_color)
            gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        v = vis.Visualizer()
        v.add_points(
            "RGB Input",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        if "labels" in target_full:
            v.add_points(
                "Semantics (GT)",
                gt_pcd_pos,
                colors=gt_pcd_color,
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )
            v.add_points(
                "Instances (GT)",
                gt_pcd_pos,
                colors=gt_inst_pcd_color, 
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []

        if sorted_masks is not None:
            for did in range(len(sorted_masks)):
                instances_colors = torch.from_numpy(
                    np.vstack(
                        get_evenly_distributed_colors(
                            max(1, sorted_masks[did].shape[1])
                        )
                    )
                )

                for i in reversed(range(sorted_masks[did].shape[1])):
                    coords = full_res_coords[
                        sorted_masks[did][:, i].astype(bool), :
                    ]

                    mask_coords = full_res_coords[
                        sorted_masks[did][:, i].astype(bool), :
                    ]
                    mask_normals = original_normals[
                        sorted_masks[did][:, i].astype(bool), :
                    ]

                    label = sort_classes[did][i]

                    if len(mask_coords) == 0:
                        continue

                    pred_coords.append(mask_coords)
                    pred_normals.append(mask_normals)

                    pred_sem_color.append(
                        self.validation_dataset.map2color([label]).repeat(
                            mask_coords.shape[0], 1
                        )
                    )

                    pred_inst_color.append(
                        instances_colors[i % len(instances_colors)]
                        .unsqueeze(0)
                        .repeat(mask_coords.shape[0], 1)
                    )


                if len(pred_coords) > 0:
                    pred_coords = np.concatenate(pred_coords)
                    pred_normals = np.concatenate(pred_normals)
                    pred_sem_color = np.concatenate(pred_sem_color)
                    pred_inst_color = np.concatenate(pred_inst_color)

                    v.add_points(
                        "Semantics (Mask3D)",
                        pred_coords,
                        colors=pred_sem_color,
                        normals=pred_normals,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )
                    v.add_points(
                        "Instances (Mask3D)",
                        pred_coords,
                        colors=pred_inst_color,
                        normals=pred_normals,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )

        out_json = []
        valid_mask_count = 0
        if query_text is not None:
            if isinstance(query_text, np.ndarray):
                query_text = query_text.tolist()
            for index, query in enumerate(query_text):
                if not query_mask[index].any():
                    continue
                valid_mask_count += 1
                if valid_mask_count > max_num_of_queries:
                    break
                use_color = np.array([255, 0, 0])[:, np.newaxis] 
                gt_use_color = np.array([0, 255, 0])[:, np.newaxis] 

                out_json.append({"text": query,
                                 "name": f"Query text {index}",
                                 "inst_coordscore": query_mask_instance_coordscore[index].tolist()
                                 }
                                )

                intersection = query_mask[index] & gt_query_mask[index]
                union = query_mask[index] | gt_query_mask[index]
                iou = intersection.sum() / (union.sum() + 1)

                union_coords = full_res_coords[union.astype(bool), :]
                union_normals = original_normals[union.astype(bool), :]
                use_color = use_color.repeat(full_res_coords.shape[0], 1).T
                use_color[gt_query_mask[index]] = gt_use_color.T 
                use_color[intersection] = np.asarray([[255, 255, 0]]) 

                v.add_points(
                    f"Query: text {index}",  
                    union_coords,
                    colors=use_color,
                    normals=union_normals,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )

            import json
            from json import encoder
            encoder.FLOAT_REPR = lambda o: format(o, '.2f')
            os.makedirs(
                f"{self.config['general']['save_dir']}/visualizations", exist_ok=True)
            if len(out_json) > 0:
                json.dump(out_json,
                          open(
                              f"{self.config['general']['save_dir']}/visualizations/{file_name}query.json", 'w'),
                          indent=4
                          )

        v.save(
            f"{self.config['general']['save_dir']}/visualizations/{file_name}"
        )

    def save_taskscheduling_visualizations(
        self,
        llm_out_json, 
        pred_inst_masks, 
        target_full, 
        full_res_coords, 
        file_name,
        original_colors,
        original_normals,
        raw_masks,
        point_size=20,
        max_num_of_queries=200,
    ):
        
        full_res_coords -= full_res_coords.mean(axis=0)

        v = vis.Visualizer()
        v.add_points(
            "RGB Input",
            full_res_coords,
            colors=original_colors,
            visible=True,
            point_size=point_size,
        )

        out_json = []
        valid_mask_count = 0
        for q_id, instance in enumerate(llm_out_json):
            if "taskscheduling" in instance["type"]:
                out_json.append({"question": instance["input_language"],
                                 "thinking": instance["thinking"],
                                "name": f"Adataset_task_id: {q_id}, task_info",
                                }
                                ) 

                pred_info = instance["tasksheduling_result"]["pred_results_idx_list"] 
                
                
                for action_id, action_dict in enumerate(pred_info):
                    pred_id = [action_dict["pred_id"]]
                    gt_id = [action_dict["gt_label"]] 
                    text = action_dict["text"]

                    gt_mask = target_full[0]['masks'].numpy()[gt_id].sum(axis=0)>0
                    pred_mask = (raw_masks[:, pred_id].cpu().numpy().sum(axis=1) > 0)
                    use_color = np.array([255, 0, 0])[:, np.newaxis] 
                    gt_use_color = np.array([0, 255, 0])[:, np.newaxis] 

                    out_json.append({"text": text,
                                    "name": f"task_id: {q_id}, action_id: {action_id}",
                                    }
                                    )

                    intersection = pred_mask & gt_mask
                    union = pred_mask | gt_mask
                    iou = intersection.sum() / (union.sum() + 1)

                    union_coords = full_res_coords[union.astype(bool), :]
                    

                    use_color = use_color.repeat(full_res_coords.shape[0], 1).T
                    gt_coords = full_res_coords[gt_mask.astype(bool), :]
                    use_color_gt = use_color.copy()
                    use_color_gt[gt_mask] = gt_use_color.T 
                    use_color_gt = use_color_gt[gt_mask.astype(bool)]


                    use_color[gt_mask] = gt_use_color.T 
                    use_color[intersection] = np.asarray([[255, 255, 0]]) 
                    use_color = use_color[union.astype(bool)]

                    v.add_points(
                        f"task_{q_id}: action_{action_id}", 
                        union_coords,
                        colors=use_color,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )
                    v.add_points(
                        f"task_{q_id}: action_{action_id}_gt",  
                        gt_coords,
                        colors=use_color_gt,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )


                gt_action_info = instance["actions"] 
                gt_action_labels = instance["tasksheduling_result"]["action_gt_labels"] 
                assert len(gt_action_labels) == len(gt_action_info)
                for action_id, action_dict in enumerate(gt_action_info):
                    gt_id = [gt_action_labels[action_id]]
                    action_text = action_dict["action"]

                    gt_mask = target_full[0]['masks'].numpy()[gt_id].sum(axis=0)>0
                    use_color = np.array([255, 0, 0])[:, np.newaxis] 
                    gt_use_color = np.array([0, 255, 0])[:, np.newaxis] 

                    out_json.append({"text": action_text,
                                    "name": f"Adataset_task_id: {q_id}, dataset_action_id: {action_id}",
                                    }
                                    )

                

                    use_color = use_color.repeat(full_res_coords.shape[0], 1).T
                    gt_coords = full_res_coords[gt_mask.astype(bool), :]
                    use_color_gt = use_color.copy()
                    use_color_gt[gt_mask] = gt_use_color.T 
                    use_color_gt = use_color_gt[gt_mask.astype(bool)]

                    v.add_points(
                        f"Adataset_task_{q_id}: dataset_action_{action_id}",  
                        gt_coords,
                        colors=use_color_gt,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )
            else:
                pass

        import json
        from json import encoder
        encoder.FLOAT_REPR = lambda o: format(o, '.2f')
        os.makedirs(
            f"{self.config['general']['save_dir']}/taskscheduling_visualizations", exist_ok=True)
        if len(out_json) > 0:
            json.dump(out_json,
                        open(
                            f"{self.config['general']['save_dir']}/taskscheduling_visualizations/{file_name}query.json", 'w'),
                        indent=4
                        )

        v.save(
            f"{self.config['general']['save_dir']}/taskscheduling_visualizations/{file_name}"
        )


    def eval_step(self, batch, batch_idx):
        raw_data, target, file_names = batch
        inverse_maps = raw_data.inverse_maps
        target_full = raw_data.target_full
        original_colors = raw_data.original_colors
        data_idx = raw_data.idx
        original_normals = raw_data.original_normals 
        original_coordinates = raw_data.original_coordinates

        if len(raw_data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = raw_data.features[:, -3:]
            raw_data.features = raw_data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=raw_data.coordinates,
            features=raw_data.features,
            device=self.device,
        )
        import time

        start = time.perf_counter()

        output = self.forward(
            data,
            point2segment=[
                target[i]["point2segment"] for i in range(len(target))
            ],
            raw_coordinates=raw_coordinates,
            extra_lang=raw_data.extra_lang,
            is_eval=True,
        )
        end = time.perf_counter()

        output['raw_coordinates'] = raw_coordinates

        if raw_data.extra_lang is not None:
            output['extra_lang'] = raw_data.extra_lang
            if 'aux_outputs' in output:
                for aux_outputs in output["aux_outputs"]:
                    aux_outputs['extra_lang'] = raw_data.extra_lang

        losses = {}
        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            losses = self.criterion(
                output, target, mask_type=self.mask_type
            )
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.llama_config.enable_llm:
            batch_lang_infos, batch_map_target_to_query = \
                self.prepare_llm(output, raw_data.extra_lang,
                                 None, target, raw_data, file_names=file_names)

            batch_gt_inst_ids = [((i.batch_idx, i, i.max_gt_iou) if not self.training else (
                i.batch_idx, i)) for i in batch_lang_infos]
            batch_input_texts = [i.question for i in batch_lang_infos]
            batch_thinking = [i.thinking for i in batch_lang_infos] 
            batch_output_texts = [i.answer for i in batch_lang_infos]
            batch_eval_types = [i.lang_type for i in batch_lang_infos]
            batch_subtasks = [i.subtasks for i in batch_lang_infos] 
            batch_actions = [i.actions for i in batch_lang_infos]
            batch_instance_queries_hidden_state = [
                i.query_hidden_feature for i in batch_lang_infos]
            batch_instance_queries_normalized_embed = [
                i.query_normalized_embed for i in batch_lang_infos]

            save_features_for_demo = False
            if save_features_for_demo:
                saved_scene_feature = {}
                os.makedirs(
                    f'./saved/{self.config.general.experiment_name}/scene_features', exist_ok=True)
                for i, scene_id in enumerate(file_names):
                    saved_scene_feature["instance_queries_hidden_state"] = batch_instance_queries_hidden_state[i]
                    saved_scene_feature["instance_queries_normalized_embed"] = batch_instance_queries_normalized_embed[i]
                    torch.save(
                        saved_scene_feature,
                        f'./saved/{self.config.general.experiment_name}/scene_features/{scene_id}.bin',
                    )

            self.llama_model.eval()
            out_json, llm_logits = self.llama_model.evaluate(input_text_list=batch_input_texts,
                                                             batch_instance_queries_hidden_state=batch_instance_queries_hidden_state,
                                                             batch_instance_queries_normalized_embed=batch_instance_queries_normalized_embed,
                                                             use_mini_batch=self.llama_config.use_mini_batch, 
                                                             mini_batch_size=self.llama_config.test_batch_size, 
                                                             batch_out_text=batch_output_texts,
                                                             batch_eval_types=batch_eval_types,
                                                             batch_thinking=batch_thinking, 
                                                             batch_gt_inst_ids=batch_gt_inst_ids,
                                                             output_logits=True,
                                                             max_new_tokens=self.llama_config.max_new_tokens,
                                                             )
            
            # ======================== prepare llm det =============================
            assert len(target) == 1
            pred_inst_masks = (
                output["pred_masks"][0][target[0]["point2segment"].cpu()] > 0.).float().cpu().clone()
            pred_inst_masks = self.get_full_res_mask(
                pred_inst_masks, inverse_maps[0], target_full[0]['point2segment'])
            pred_inst_masks = np.array(pred_inst_masks).astype(bool)

            all_llm_dectection = []
            assert len(file_names) == 1
            last_test_type = -1
            for iinstance in llm_logits:
                # TODO: here we only test first token for one-to-many case
                if last_test_type == iinstance[0]:
                    continue
                if iinstance[0] == 0:
                    last_test_type = 0
                if last_test_type < iinstance[0]:
                    for _ in range(last_test_type+1, iinstance[0]):
                        all_llm_dectection.append(torch.zeros((1, 100)))
                all_llm_dectection.append(iinstance[1].to("cpu"))
                last_test_type = iinstance[0]
            if len(all_llm_dectection) < 198: 
                for _ in range(len(all_llm_dectection), 198):
                    all_llm_dectection.append(torch.zeros((1, 100)))
            all_llm_dectection = [torch.zeros((1, 100))] + [all_llm_dectection[0]] + [
                torch.zeros((1, 100))] + all_llm_dectection[1:]
            assert len(all_llm_dectection) == 200
            all_llm_dectection = torch.vstack(all_llm_dectection).T
            get_max = torch.max(all_llm_dectection, dim=1)
            np.savez_compressed(
                f"{self.llama_config.save_path}/{file_names[0]}.npz",
                pred_masks=pred_inst_masks,
                pred_scores=get_max[0],
                pred_classes=self.validation_dataset._remap_model_output(
                    get_max[1]),
            )

            for item, gt, evaluation_type, subtasks, actions, gt_ids, thinking in zip(out_json, batch_output_texts, batch_eval_types, batch_subtasks, batch_actions, batch_gt_inst_ids, batch_thinking):
                if item["gt"] == "NONE":
                    item["gt"] = gt
                if thinking:
                    item["thinking"] = thinking 
                item["type"] = evaluation_type
                if subtasks:
                    item["subtasks"] = subtasks
                if actions:
                    item["actions"] = actions

        all_raw_pred_instance_masks = self.eval_instance_step(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            extra_lang=raw_data.extra_lang
        )

        if self.llama_config.enable_llm:
            assert len(target) == 1
            pred_inst_masks = (
                output["pred_masks"][0][target[0]["point2segment"].cpu()] > 0.).float().cpu().clone()
            pred_inst_masks = self.get_full_res_mask(
                pred_inst_masks, inverse_maps[0], target_full[0]['point2segment'])
            pred_inst_masks = [pred_inst_masks]


            try:
                map_target_to_query, valid_target = batch_map_target_to_query[0]

                inter = pred_inst_masks[0].to(
                    bool)[:, map_target_to_query[valid_target]].T & target_full[0]['masks']
                outer = pred_inst_masks[0].to(
                    bool)[:, map_target_to_query[valid_target]].T | target_full[0]['masks']
                instance_iou = inter.sum(1) / outer.sum(1)
                out_json, score, bbox_score_25, bbox_score_50, mask_score_25, mask_score_50, m3dref_bbox_result= eval_llm_iou_score(out_json, {"pred_inst_masks": pred_inst_masks,
                                                                                                                                                "target_full": target_full,
                                                                                                                                                "batch_gt_inst_ids": batch_gt_inst_ids,
                                                                                                                                                "original_coordinates": original_coordinates
                                                                                                                                                })
            

            except Exception as e:
                return 0
        





            if self.config.general.save_taskscheduling_visualizations:
                self.save_taskscheduling_visualizations(llm_out_json=out_json, 
                                                        pred_inst_masks=pred_inst_masks, 
                                                        target_full=target_full, 
                                                        full_res_coords=original_coordinates[0], 
                                                        file_name=file_names[0], 
                                                        original_colors=original_colors[0], 
                                                        original_normals=original_normals[0], 
                                                        raw_masks = all_raw_pred_instance_masks[0],
                                                        point_size=20,
                                                        )
            with open(f"{self.llama_config.save_path}/{file_names[0]}.json", 'w') as json_file:
                json.dump({"prediction": out_json, 
                            "score": score,
                            "bbox_score_25": bbox_score_25,
                            "bbox_score_50": bbox_score_50,
                            "mask_score_25": mask_score_25,
                            "mask_score_50": mask_score_50,
                            "seg_score": instance_iou.tolist()},
                            json_file, indent=4)
            if m3dref_bbox_result:
                with open(f"{self.llama_config.save_path}/m3drefer/{file_names[0]}.pkl", 'wb') as f:
                    pickle.dump(m3dref_bbox_result, f)


        if self.config.data.test_mode != "test":
            return {
                f"val_{k}": v.detach().cpu().item() for k, v in losses.items()
            }
        else:
            return 0.0

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                self.config.general.topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = torch.div(topk_indices, torch.tensor(
            num_classes), rounding_mode='floor')  # class share the same mask
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords, 
        original_colors,
        original_normals,
        raw_coords,
        idx,
        extra_lang=None,
    ):
        label_offset = self.validation_dataset.label_offset
        if 'aux_outputs' in output:
            prediction = output["aux_outputs"]
        else:
            print('No aux outputs are found.')
            prediction = []
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        assert self.config.model.num_classes - \
            1 == self.config.data.num_labels - label_offset
        pred_lang_logits = []
        if self.config.model.language_model and not self.config.model.softmax_mode:
            pred_logits = []
            for pred_logit in prediction[self.decoder_id]["pred_logits"]:
                if not self.config.data.sample_class_labels or not self.training:
                    pred_logits.append(
                        pred_logit[:, :self.config.model.num_classes - 1].sigmoid())
                    pred_lang_logits.append(
                        pred_logit[:, self.config.model.num_classes - 1:].sigmoid())
                else:
                    pred_lang_logits.append(pred_logit.sigmoid())

            if not self.config.data.sample_class_labels or not self.training:
                prediction[self.decoder_id][
                    "pred_logits"
                ] = torch.stack(pred_logits, dim=0)
            prediction[self.decoder_id][
                "pred_lang_logits"
            ] = pred_lang_logits
        elif not self.config.model.language_model and not self.config.model.softmax_mode:
            if isinstance(prediction[self.decoder_id]["pred_logits"], list):
                prediction[self.decoder_id]["pred_logits"] = torch.stack(
                    prediction[self.decoder_id]["pred_logits"], 0)
            prediction[self.decoder_id][
                "pred_logits"
            ] = prediction[self.decoder_id]["pred_logits"].sigmoid()
        else:
            assert not self.config.data.sample_class_labels
            if isinstance(prediction[self.decoder_id]["pred_logits"], list):
                prediction[self.decoder_id]["pred_logits"] = torch.stack(
                    prediction[self.decoder_id]["pred_logits"], 0)
            prediction[self.decoder_id][
                "pred_logits"
            ] = torch.functional.F.softmax(
                prediction[self.decoder_id]["pred_logits"], dim=-1
            )[
                ..., :-1
            ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()

        all_extra_query_texts = list()
        all_pred_extra_masks = list()
        all_pred_extra_masks_instance_coordscore = list()
        all_gt_extra_masks = list()
        all_gt_ious = []
        all_raw_pred_instance_masks = list()
        all_iou_25_f1_score = []
        all_iou_50_f1_score = []

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])): 
            if self.model.train_on_segments:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()[target_low_res[bid]["point2segment"].cpu()]
                )  
            else:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )
            if not self.config.data.sample_class_labels or not self.training:
                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                        "pred_lang_logits": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[
                        offset_coords_idx: curr_coords_idx + offset_coords_idx
                    ]
                    offset_coords_idx += curr_coords_idx

                    # for each query in num_queries
                    for curr_query in range(masks.shape[1]):
                        # [num_points, query_i]
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(  # current mask divided into cluster
                                        original_pred_masks
                                        * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(  # copy score
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )
                                    if len(pred_lang_logits) > 0:
                                        new_preds["pred_lang_logits"].append(  # copy score
                                            prediction[self.decoder_id][
                                                "pred_lang_logits"
                                            ][bid][curr_query]
                                        )

                    if len(pred_lang_logits) > 0:
                        new_masks = torch.stack(new_preds["pred_masks"], dim=1)

                        # for computing (num_points, num_query)
                        raw_masks = (new_masks > 0.).float().cpu().clone()
                        raw_heatmap = new_masks.float().cpu().clone()
                        raw_masks = self.get_full_res_mask(
                            raw_masks, inverse_maps[bid], target_full_res[bid]['point2segment'])
                        raw_heatmap = self.get_full_res_mask(
                            raw_heatmap, inverse_maps[bid], target_full_res[bid]['point2segment'], is_heatmap=True)
                        if len(new_preds["pred_lang_logits"]) > 0:
                            pred_lang_logits = torch.stack(
                                new_preds["pred_lang_logits"])
                        else:
                            pred_lang_logits = torch.zeros(
                                (0, prediction[self.decoder_id]["pred_lang_logits"][bid].shape[1]), dtype=torch.float32, device='cuda')
                        prediction[self.decoder_id]['pred_lang_logits'][bid] = pred_lang_logits
                    else:
                        raw_masks = None

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                    )
                else:
                    # # for computing (num_points, num_query)
                    raw_masks = (masks > 0.).float().cpu().clone()
                    raw_heatmap = masks.float().cpu().clone()
                    raw_masks = self.get_full_res_mask(
                        raw_masks, inverse_maps[bid], target_full_res[bid]['point2segment'])
                    raw_heatmap = self.get_full_res_mask(
                        raw_heatmap, inverse_maps[bid], target_full_res[bid]['point2segment'], is_heatmap=True)

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes - 1,
                    )

            all_raw_pred_instance_masks.append(raw_masks)

            if not self.config.data.sample_class_labels or not self.training:
                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

            if not self.config.data.sample_class_labels or not self.training:
                masks = masks.numpy()
                heatmap = heatmap.numpy()

                sort_scores = scores.sort(descending=True)
                sort_scores_index = sort_scores.indices.cpu().numpy()
                sort_scores_values = sort_scores.values.cpu().numpy()
                sort_classes = classes[sort_scores_index]

                sorted_masks = masks[:, sort_scores_index]
                sorted_heatmap = heatmap[:, sort_scores_index]

            if not self.config.data.sample_class_labels or not self.training:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

            if len(extra_lang) > 0 and self.config.model.language_model:
                gt_ious, gt_extra_masks, extra_query_texts, pred_extra_masks, pred_extra_masks_instance_coordscore = eval_seg_model(bid=bid,
                                                                                                                                    config=self.config,
                                                                                                                                    extra_lang=extra_lang,
                                                                                                                                    full_res_coords=full_res_coords,
                                                                                                                                    raw_masks=raw_masks,
                                                                                                                                    raw_heatmap=raw_heatmap,
                                                                                                                                    target_full_res=target_full_res,
                                                                                                                                    pred_lang_logits=prediction[self.decoder_id][
                                                                                                                                        'pred_lang_logits'][bid],
                                                                                                                                    training=self.training,
                                                                                                                                    )
                all_gt_ious.append(gt_ious)
                all_gt_extra_masks.append(gt_extra_masks)
                all_extra_query_texts.append(extra_query_texts)
                all_pred_extra_masks.append(pred_extra_masks)
                all_pred_extra_masks_instance_coordscore.append(
                    pred_extra_masks_instance_coordscore)

        if self.validation_dataset.dataset_name == "scannet200":
            # remap gt labels
            # this code originally is out of the bid loop, which seems to be a bug.
            for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                if self.config.data.test_mode != "test":
                    target_full_res[bid]["labels"][
                        target_full_res[bid]["labels"] == 0
                    ] = -1

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                # GT BOX
                bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][
                        target_full_res[bid]["masks"][obj_id, :]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )

                self.bbox_gt[file_names[bid]] = bbox_data

        if not self.config.data.sample_class_labels or not self.training:
            if self.validation_dataset.dataset_name == "scannet200":
                all_pred_classes[bid][all_pred_classes[bid] == 0] = -1

            for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                all_pred_classes[
                    bid
                ] = self.validation_dataset._remap_model_output(
                    all_pred_classes[bid].cpu() + label_offset
                )

                if (
                    self.config.data.test_mode != "test"
                    and len(target_full_res) != 0
                ):
                    bbox_data = []
                    for query_id in range(
                        all_pred_masks[bid].shape[1]
                    ):  # self.model.num_queries
                        obj_coords = full_res_coords[bid][
                            all_pred_masks[bid][:, query_id].astype(bool), :
                        ]
                        if obj_coords.shape[0] > 0:
                            obj_center = obj_coords.mean(axis=0)
                            obj_axis_length = obj_coords.max(
                                axis=0
                            ) - obj_coords.min(axis=0)

                            bbox = np.concatenate(
                                (obj_center, obj_axis_length))

                            bbox_data.append(
                                (
                                    all_pred_classes[bid][query_id].item(),
                                    bbox,
                                    all_pred_scores[bid][query_id],
                                )
                            )
                    self.bbox_preds[file_names[bid]] = bbox_data

                self.preds[file_names[bid]] = {
                    "pred_masks": (all_pred_masks[bid]).astype(bool),
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                    'gt_ious': all_gt_ious[bid] if len(all_gt_ious) > 0 else (np.zeros((0,), dtype=float), np.zeros((0,), dtype=str), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)),
                }
                if self.config.general.export:
                    self.export(
                        self.preds[file_names[bid]]["pred_masks"],
                        self.preds[file_names[bid]]["pred_scores"],
                        self.preds[file_names[bid]]["pred_classes"],
                        file_names[bid],
                        self.decoder_id,
                    )

                if 'gt_ious' in self.preds[file_names[bid]]:
                    gt_ious = self.preds[file_names[bid]]['gt_ious'][0]
                    multi_iou_25_f1_score = self.preds[file_names[bid]
                                                       ]['gt_ious'][3]
                    multi_iou_50_f1_score = self.preds[file_names[bid]
                                                       ]['gt_ious'][4]
                    assert len(multi_iou_25_f1_score) == len(
                        multi_iou_50_f1_score) == len(gt_ious)


                if self.config.general.gpus > 1:
                    dump_bbox = [self.bbox_preds[file_names[bid]],
                                 self.bbox_gt[file_names[bid]]]
                    with open(osp.join(self.tmpdir, file_names[bid] + "_bbox.pkl"), 'wb') as f:
                        pickle.dump(dump_bbox, f, protocol=2)

                    np.savez_compressed(osp.join(self.tmpdir, file_names[bid] + '_preds.npz'),
                                        pred_masks=self.preds[file_names[bid]]['pred_masks'].astype(
                                            bool),
                                        pred_scores=self.preds[file_names[bid]
                                                               ]['pred_scores'],
                                        pred_classes=self.preds[file_names[bid]
                                                                ]['pred_classes'],
                                        gt_ious=self.preds[file_names[bid]
                                                           ]['gt_ious'] if 'gt_ious' in self.preds[file_names[bid]] else None,
                                        )

                    torch.distributed.barrier()

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if self.config.general.save_visualizations:
                self.save_visualizations(
                    target_full_res[bid], 
                    full_res_coords[bid], 
                    [self.preds[file_names[bid]]["pred_masks"]
                     ] if not self.config.data.sample_class_labels or not self.training else None,
                    [self.preds[file_names[bid]]["pred_classes"]
                     ] if not self.config.data.sample_class_labels or not self.training else None,
                    file_names[bid],
                    original_colors[bid],
                    original_normals[bid],
                    [self.preds[file_names[bid]]["pred_scores"]
                     ] if not self.config.data.sample_class_labels or not self.training else None,
                    point_size=self.config.general.visualization_point_size,
                    query_text=all_extra_query_texts[bid] if len(
                        extra_lang) > 0 else None,
                    query_mask=all_pred_extra_masks[bid] if len(
                        extra_lang) > 0 else None,
                    gt_query_mask=all_gt_extra_masks[bid] if len(
                        extra_lang) > 0 else None,
                    query_mask_instance_coordscore=all_pred_extra_masks_instance_coordscore[bid] if len(
                        extra_lang) > 0 else None,
                )

        return all_raw_pred_instance_masks

    def eval_instance_epoch_end(self, all_preds, all_bbox_preds, all_bbox_gt):
        log_prefix = f"val"
        ap_results = {}

        head_results, tail_results, common_results = [], [], []

        box_ap_50 = eval_det(
            all_bbox_preds, all_bbox_gt, ovthresh=0.5, use_07_metric=False
        )
        box_ap_25 = eval_det(
            all_bbox_preds, all_bbox_gt, ovthresh=0.25, use_07_metric=False
        )
        mean_box_ap_25 = sum([v for k, v in box_ap_25[-1].items()]) / len(
            box_ap_25[-1].keys()
        )
        mean_box_ap_50 = sum([v for k, v in box_ap_50[-1].items()]) / len(
            box_ap_50[-1].keys()
        )

        ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50

        for class_id in box_ap_50[-1].keys():
            try:
                class_name = self.train_dataset.label_info[class_id]["name"]
                ap_results[f"{log_prefix}_{class_name}_val_box_ap_50"] = box_ap_50[
                    -1
                ][class_id]
            except Exception as e:
                print(e)
                class_name = 'invalid'
                continue

        for class_id in box_ap_25[-1].keys():
            try:
                class_name = self.train_dataset.label_info[class_id]["name"]
                ap_results[f"{log_prefix}_{class_name}_val_box_ap_25"] = box_ap_25[
                    -1
                ][class_id]
            except Exception as e:
                print(e)
                class_name = 'invalid'
                continue

        base_path = f"saved/eval_output/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}"

        pred_path = f"{base_path}/tmp_output.txt"

        log_prefix = f"val"

        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        try:
            if self.validation_dataset.dataset_name in ["scannet","scannet200"]:
                gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
            else:
                gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"
            if self.validation_dataset.dataset_name in ["scannet","scannet200",]:
                evaluate(
                    all_preds,
                    gt_data_path,
                    pred_path,
                    dataset=self.validation_dataset.dataset_name,
                ) 

            with open(pred_path, "r") as fin:
                for line_id, line in enumerate(fin):
                    if line_id == 0:
                        continue
                    class_name, _, ap, ap_50, ap_25 = line.strip().split(",")

                    if self.validation_dataset.dataset_name == "scannet200":
                        if class_name in VALID_CLASS_IDS_200_VALIDATION:
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap"
                            ] = float(ap)
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap_50"
                            ] = float(ap_50)
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap_25"
                            ] = float(ap_25)

                            if class_name in HEAD_CATS_SCANNET_200:
                                head_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            elif class_name in COMMON_CATS_SCANNET_200:
                                common_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            elif class_name in TAIL_CATS_SCANNET_200:
                                tail_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            else:
                                raise ValueError("class not known!")
                    else:
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap"
                        ] = float(ap)
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap_50"
                        ] = float(ap_50)
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap_25"
                        ] = float(ap_25)

            if self.validation_dataset.dataset_name == "scannet200":
                head_results = np.stack(head_results)
                common_results = np.stack(common_results)
                tail_results = np.stack(tail_results)

                mean_tail_results = np.nanmean(tail_results, axis=0)
                mean_common_results = np.nanmean(common_results, axis=0)
                mean_head_results = np.nanmean(head_results, axis=0)

                ap_results[
                    f"{log_prefix}_mean_tail_ap_25"
                ] = mean_tail_results[0]
                ap_results[
                    f"{log_prefix}_mean_common_ap_25"
                ] = mean_common_results[0]
                ap_results[
                    f"{log_prefix}_mean_head_ap_25"
                ] = mean_head_results[0]

                ap_results[
                    f"{log_prefix}_mean_tail_ap_50"
                ] = mean_tail_results[1]
                ap_results[
                    f"{log_prefix}_mean_common_ap_50"
                ] = mean_common_results[1]
                ap_results[
                    f"{log_prefix}_mean_head_ap_50"
                ] = mean_head_results[1]

                ap_results[
                    f"{log_prefix}_mean_tail_ap_25"
                ] = mean_tail_results[2]
                ap_results[
                    f"{log_prefix}_mean_common_ap_25"
                ] = mean_common_results[2]
                ap_results[
                    f"{log_prefix}_mean_head_ap_25"
                ] = mean_head_results[2]

                overall_ap_results = np.nanmean(
                    np.vstack((head_results, common_results, tail_results)),
                    axis=0,
                )

                ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
                ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
                ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]

                ap_results = {
                    key: 0.0 if math.isnan(score) else score
                    for key, score in ap_results.items()
                }
            else:
                mean_ap = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap")
                    ]
                )
                mean_ap_50 = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap_50")
                    ]
                )
                mean_ap_25 = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap_25")
                    ]
                )

                ap_results[f"{log_prefix}_mean_ap"] = mean_ap
                ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
                ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

                ap_results = {
                    key: 0.0 if math.isnan(score) else score
                    for key, score in ap_results.items()
                }
        except (IndexError, OSError,AttributeError) as e:
            print("NO SCORES!!!")
            ap_results[f"{log_prefix}_mean_ap"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_25"] = 0.0

        ap_results = collect_grounding_score(all_preds, ap_results, log_prefix)

        with open(self.tmpdir + '/ap_results.pkl', 'wb') as f:
            pickle.dump(ap_results, f, protocol=2)

        try:
            if not self.config.general.export:
                shutil.rmtree(base_path)
        except FileNotFoundError as e:
            pass

        return ap_results

    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return
        if len(self.preds) == 0:
            print('===================== found zero prediction ===================')
            return

        if not self.config.data.sample_class_labels or not self.training:
            if self.config.general.gpus > 1:
                # clean
                del self.preds
                del self.bbox_preds
                del self.bbox_gt

                gc.collect()

                all_preds = {}
                all_bbox_preds = {}
                all_bbox_gt = {}
                for i in glob.glob(self.tmpdir + '/*_preds.npz'):
                    data = np.load(i, mmap_mode='r')
                    scene_name = i.split('/')[-1].split('_preds.npz')[0]
                    all_preds[scene_name] = dict(
                        pred_masks=data['pred_masks'], pred_classes=data['pred_classes'], pred_scores=data['pred_scores'], gt_ious=data['gt_ious'])
                    with open(i.replace('_preds.npz', '_bbox.pkl'), 'rb') as f:
                        all_bbox_preds[scene_name], all_bbox_gt[scene_name] = pickle.load(
                            f)

                if self.global_rank == 0:
                    self.eval_instance_epoch_end(
                        all_preds, all_bbox_preds, all_bbox_gt)

                torch.distributed.barrier()
                with open(self.tmpdir + '/ap_results.pkl', 'rb') as f:
                    ap_results = pickle.load(f)

                if self.global_rank == 0:
                    for i in glob.glob(self.tmpdir + '/*_preds.npz'):
                        os.remove(i)
                        os.remove(i.replace('_preds.npz', '_bbox.pkl'))
            else:
                all_preds = self.preds
                all_bbox_preds = self.bbox_preds
                all_bbox_gt = self.bbox_gt
                ap_results = self.eval_instance_epoch_end(
                    all_preds, all_bbox_preds, all_bbox_gt)

                del self.preds
                del self.bbox_preds
                del self.bbox_gt

                gc.collect()

            ap_results = {k: v for k, v in ap_results.items()
                          if k.startswith('val_mean')}
            self.log_dict(ap_results)
            print({k: f'{v:.4f}' for k, v in ap_results.items()})

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        def gather_cpu(obj, tmpdir, rank, total_rank):
            with open(tmpdir + f'{rank}.pkl', 'wb') as f:
                pickle.dump(obj, f, protocol=2)
            torch.distributed.barrier()
            objs = []
            for i in range(total_rank):
                with open(tmpdir + f'{rank}.pkl', 'rb') as f:
                    objs.append(pickle.load(f))
            return objs

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():
                dd[key].append(val)

        if self.config.general.gpus > 1:
            # sync multi gpu
            dd_mgpu = gather_cpu(dd, self.tmpdir + '/losses',
                                 self.global_rank, self.config.general.gpus)
            dd = {k: [] for k in dd_mgpu[0]}
            for bd in dd_mgpu:
                for k in dd:
                    dd[k].extend(bd[k])

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd["val_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )

        self.log_dict(dd)

        print(self.config.general.experiment_name)

    def configure_optimizers(self):
        other_params = [p for n, p in self.named_parameters()
                        if 'language_model' not in n]
        lang_params = [p for n, p in self.named_parameters()
                       if 'language_model' in n]
        weight_decay = 1e-4

        params = [
            {'params': other_params, "lr": self.config.optimizer.lr,
                "weight_decay": weight_decay},
            {'params': lang_params, "lr": self.config.optimizer.lr *
                0.1, "weight_decay": weight_decay},
        ]

        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=params
        )

        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        if self.config.general.gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True)
            self.config.data.train_dataloader.shuffle = False
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
            sampler=sampler if self.config.general.gpus > 1 else None
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        if self.config.general.gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.validation_dataset, shuffle=False)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
            sampler=sampler if self.config.general.gpus > 1 else None
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        if self.config.general.gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset, shuffle=False)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
            sampler=sampler if self.config.general.gpus > 1 else None
        )
