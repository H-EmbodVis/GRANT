from cProfile import label
import csv
import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform, shuffle
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from random import randrange
import json
import os

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

import torch
from torch.utils.data import ConcatDataset

from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
    CLASS_LABELS_200,
    CLASS_LABELS_20,
    VALID_CLASS_IDS_200
)
from datasets.utils import read_axis_align_matrix, concatenate_texts_with_separator

from datasets.language_info import lang_info_data, grounding_data
from datasets.data_aug import *
from conf.paths import BERT_PATH


class SceneVerseBase(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_name="scannet", 
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet", 
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        task="instance_segmentation",
        filter_out_classes=[],
        label_offset=0,
        is_elastic_distortion=True,
        lang_query=False,
        positive_lang_query_ratio=0.5,
        lang_max_token_length=256,
        num_concat_texts=4,
        bert_path=BERT_PATH,
        lang_data_conf='',
        sample_class_labels=False,
        axis_align_coord=False,
        scenes=None,
    ):
        assert task in [
            "instance_segmentation",
        ], "unknown task"

        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.sample_class_labels = sample_class_labels

        self.lang_query = lang_query
        self.positive_lang_query_ratio = positive_lang_query_ratio
        self.num_concat_texts = num_concat_texts
        self.axis_align_coord = axis_align_coord

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
            self.color_map[255] = (255, 255, 255)
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_raw_coordinates = add_raw_coordinates
        self.lang_data_conf = lang_data_conf

        # loading database files
        self._data = scenes
        labels = self._load_yaml(Path(label_db_filepath))

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            raise ValueError(
                "pass mean and std as tuple of tuples, or as an .yaml file"
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.scene_ids = set(scenes)
        self.sceneverse_dataset_name = data_dir.split('/')[-1]

        self.lang_max_token_length = lang_max_token_length
        if self.num_concat_texts > 0:
            from transformers import AutoTokenizer, BertConfig
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_path, model_max_length=self.lang_max_token_length)

        if self.dataset_name == 'scannet':
            self.dataset_class_labels = CLASS_LABELS_20
        elif self.dataset_name == 'scannet200':
            self.dataset_class_labels = CLASS_LABELS_200
        else:
            raise NotImplementedError


        for k in lang_data_conf.split('+'):
            k = k.split(',')[0]
            assert k in ['scanrefer', 'm3dref', 'groundedscenecaption', 'scan2cap', 'scanqa', 'objdesc',
                         'scenedesc', '3dllm', 'alpaca', 'none', 'embodieddialog', 'embodiedplan', "globalscenecap", "noscanrefer", "taskrefer", "taskscheduling"]

        if self.lang_query > 0:
            self.multi_lang_source = []
            if self.sceneverse_dataset_name == 'ScanNet':
                if 'scanrefer' in lang_data_conf:
                    with open('./data/langdata/scanrefer_format.json') as f:
                        scanrefer_source = json.load(f)
                    scanrefer_source = [
                        i for i in scanrefer_source if i['scene_id'] in self.scene_ids]
                    self.multi_lang_source.extend(scanrefer_source)
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added ScanRefer Database: {len(scanrefer_source)}')

                if 'm3dref' in lang_data_conf:
                    with open('./data/langdata/m3dref_format.json') as f:
                        m3dref_source = json.load(f)
                    m3dref_source = [
                        i for i in m3dref_source if i['scene_id'] in self.scene_ids]
                    self.multi_lang_source.extend(m3dref_source)
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Multi3DRef Database: {len(m3dref_source)}')

                if 'groundedscenecaption' in lang_data_conf and self.mode == 'train': 
                    with open('./data/langdata/groundedscenecaption_format.json') as f:
                        GroundedSceneCaption_source = json.load(f)
                    GroundedSceneCaption_source = [
                        i for i in GroundedSceneCaption_source if i['scene_id'] in self.scene_ids]
                    self.multi_lang_source.extend(GroundedSceneCaption_source)
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Grounded Scene Caption Database: {len(GroundedSceneCaption_source)}')

            if 'taskrefer' in lang_data_conf:
                with open('./data/langdata/taskrefer_format.json') as f:
                    taskrefer_source = json.load(f)
                taskrefer_source = [
                    i for i in taskrefer_source if i['scene_id'] in self.scene_ids and i["dataset"] == self.sceneverse_dataset_name]
                self.multi_lang_source.extend(taskrefer_source)
                print(
                    f'[{self.sceneverse_dataset_name}][{self.mode}] Added TaskRefer Database: {len(taskrefer_source)}')
                
            self.multi_lang_source = [
                i for i in self.multi_lang_source if i['scene_id'] in self.scene_ids]
            self.multi_lang_dict = {}
            for i in self.multi_lang_source:
                if not i['scene_id'] in self.multi_lang_dict:
                    self.multi_lang_dict[i['scene_id']] = [i]
                else:
                    self.multi_lang_dict[i['scene_id']].append(i)
            assert set(self.multi_lang_dict.keys()).issubset(self.scene_ids)

            instruction_following_sources = []

            if self.sceneverse_dataset_name == 'ScanNet':
                if 'scanqa' in lang_data_conf:
                    with open('./data/langdata/scanqa_format.json') as f:
                        scanqa_lang_source = json.load(f)
                    scanqa_lang_source = [
                        i for i in scanqa_lang_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added ScanQA Database: {len(scanqa_lang_source)}')
                    instruction_following_sources.extend(scanqa_lang_source)

                if 'objdesc' in lang_data_conf:
                    with open('./data/langdata/objectdescription_format.json') as f:
                        objectdescription_source = json.load(f)
                    objectdescription_source = [
                        i for i in objectdescription_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Object Description dataset {len(objectdescription_source)}.')
                    instruction_following_sources.extend(objectdescription_source)

                if 'scenedesc' in lang_data_conf:

                    with open('./data/langdata/groundedscenecaption_format.json') as f:
                        scenedesc_source = json.load(f)


                    scenedesc_source = [
                        i for i in scenedesc_source if i['scene_id'] in self.scene_ids]
                    for i, lang in enumerate(scenedesc_source):
                        qa_dict = dict(
                            scene_id=lang['scene_id'],
                            answer=lang['description'],
                            object_ids=lang['object_ids'],
                            all_phrases_positions=lang['all_phrases_positions'],
                            lang_type='scenedesc:v3',
                        )
                        scenedesc_source[i] = qa_dict
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Scene Description Database: {len(scenedesc_source)}.')
                    instruction_following_sources.extend(scenedesc_source)

                if 'scan2cap' in lang_data_conf:
                    with open('./data/langdata/scanrefer_format.json') as f:
                        scan2cap_source = json.load(f)
                    scan2cap_source = [
                        i for i in scan2cap_source if i['scene_id'] in self.scene_ids]

                    for i, cap in enumerate(scan2cap_source):
                        scene_id = cap['scene_id']
                        cap['lang_type'] = 'scan2cap:' + cap['eval_type']
                        qa_dict = dict(
                            scene_id=cap['scene_id'],
                            answer=cap['description'],
                            object_ids=cap['object_ids'],
                            lang_type=cap['lang_type'],
                            all_phrases_positions=cap['all_phrases_positions']
                        )
                        scan2cap_source[i] = qa_dict

                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added scan2cap(ScanRefer) Database: {len(scan2cap_source)}')
                    instruction_following_sources.extend(scan2cap_source)

                if '3dllm' in lang_data_conf:
                    with open('./data/langdata/3dllm_format.json') as f:
                        data_3dllm_source = json.load(f)
                    data_3dllm_source = [
                        i for i in data_3dllm_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added 3D LLM dataset {len(data_3dllm_source)}.')
                    instruction_following_sources.extend(data_3dllm_source)

                if 'embodiedplan' in lang_data_conf:
                    with open('./data/langdata/embodiedplan_format.json') as f:
                        embodiedplan_source = json.load(f)
                    embodiedplan_source = [
                        i for i in embodiedplan_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Embodied Planning dataset {len(embodiedplan_source)}.')
                    instruction_following_sources.extend(embodiedplan_source)

                if 'embodieddialog' in lang_data_conf:
                    with open('./data/langdata/embodieddialog_format.json') as f:
                        embodieddialog_source = json.load(f)
                    embodieddialog_source = [
                        i for i in embodieddialog_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Embodied Dialog dataset {len(embodieddialog_source)}.')
                    instruction_following_sources.extend(embodieddialog_source)

                if 'globalscenecap' in lang_data_conf:
                    with open('./data/langdata/global_scene_cap_format.json') as f:
                        global_scene_caption_source = json.load(f)
                    global_scene_caption_source = [
                        i for i in global_scene_caption_source if i['scene_id'] in self.scene_ids]
                    print(
                        f'[{self.sceneverse_dataset_name}][{self.mode}] Added Global Caption dataset {len(global_scene_caption_source)}.')
                    instruction_following_sources.extend(
                        global_scene_caption_source)
                
            if 'taskscheduling' in lang_data_conf:
                with open('./data/langdata/ORS3D-60K.json') as f:
                    taskscheduling_source = json.load(f)
                taskscheduling_source = [
                    i for i in taskscheduling_source if i['scene_id'] in self.scene_ids and i["dataset"] == self.sceneverse_dataset_name]
                print(
                    f'[{self.sceneverse_dataset_name}][{self.mode}] Added ORS3D-60K dataset, item num: {len(taskscheduling_source)}.')
                instruction_following_sources.extend(taskscheduling_source)

            self.instruction_lang_dict = {}
            for i in instruction_following_sources:
                if not i['scene_id'] in self.instruction_lang_dict:
                    self.instruction_lang_dict[i['scene_id']] = [i]
                else:
                    self.instruction_lang_dict[i['scene_id']].append(i)

        max_sample_lang_type_count = {
            'scanqa': 10,
            'objdesc': 10,
            'scenedesc': 0,
            'scan2cap': 10,
            '3dllm': 0,
            'embodiedplan': 0,
            'embodieddialog': 0,
            "globalscenecap": 0,
            'taskscheduling': 0,
        }
        for k in lang_data_conf.split('+'):
            if ',' in k:
                lang_type, sample_num = k.split(',')
                max_sample_lang_type_count[lang_type] = int(sample_num)
        self.max_sample_lang_type_count = max_sample_lang_type_count

        # avoid empty training
        if 'nocls' in self.lang_data_conf and self.mode == 'train':
            self._data = [i for i in self._data if i in self.multi_lang_dict] 

        print(f'Finish Loading [{self.sceneverse_dataset_name}] {self.mode} scenes: {len(self._data)}')
        print('-'*50)

        self.alpaca_source = []
        if 'alpaca' in self.lang_data_conf and self.mode == 'train':
            with open("data/langdata/alpaca_data.json", 'r') as f:
                alpaca_source = json.load(f)
            print(f'[{self.sceneverse_dataset_name}][{self.mode}] Added Alpaca dataset {len(alpaca_source)}.')
            self.alpaca_source = alpaca_source
        
        sceneverse_dir = "/".join(data_dir.split('/')[:-1])
        self.int2cat = json.load(open(os.path.join(sceneverse_dir,"ScanNet",
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(sceneverse_dir,"ScanNet",
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        scan_data = self._load_scan(self.data[idx], filter_bkg=False)
        
        try:
            coordinates, color = scan_data['scene_pcds'][0], scan_data['scene_pcds'][1]
            superpoints = scan_data['scene_superpoints']
            segments = superpoints[1, :].numpy()
            labels = scan_data["labels"]
            normals = None
            scene_id = self.data[idx]
            coordinates -= coordinates.mean(0)
            raw_coordinates = coordinates.copy()
            raw_color = color
            raw_normals = normals

        except Exception as e: 
            print(f"Error loading scan data: {e}")
            return self.__getitem__(0)




        if not self.add_colors:
            color = np.ones((len(color), 3))

        if "train" in self.mode:
            coordinates += (
                np.random.uniform(coordinates.min(0), coordinates.max(0))
                / 2
            )


            for i in (0, 1):  # flip x,y planes
                if np.random.rand() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            aug = self.volume_augmentations(  # scale, rotate the scene
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )

            if np.random.rand() < 0.95:
                if float(self.is_elastic_distortion) > 0.:
                    for granularity, magnitude in ((0.2, 0.4 * float(self.is_elastic_distortion)), (0.8, 1.6 * float(self.is_elastic_distortion))):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )

            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        extra_groundings = grounding_data()

        if self.num_concat_texts > 0 and ((not 'nocls' in self.lang_data_conf) or (not self.mode == 'train')):
            if (not self.sample_class_labels or self.mode != 'train'):
                text_class_labels = list(deepcopy(self.dataset_class_labels))
                for cls_id, class_label in enumerate(text_class_labels):
                    if cls_id in self.filter_out_classes:
                        continue
                    extra_groundings.add_detection(class_label, gt_insts=np.unique(
                        labels[(labels[:, 0] == cls_id), 1]).tolist()) 
            else:
                text_class_labels = list(deepcopy(self.dataset_class_labels))

                positive_cls_id_sets = set(np.unique(labels[:, 0]))
                negative_cls_id_sets = np.asarray(
                    list(set(np.arange(len(self.dataset_class_labels))) - positive_cls_id_sets))
                np.random.shuffle(negative_cls_id_sets)
                negative_cls_id_sets = negative_cls_id_sets[:int(
                    len(positive_cls_id_sets) * (np.random.rand() * 2.))] 

                # positive labels:
                for cls_id in positive_cls_id_sets:
                    if not (0 <= cls_id < len(text_class_labels)):
                        continue  # 255 / -1 ignore
                    if cls_id in self.filter_out_classes:
                        continue  # continue rather concat
                    class_label = text_class_labels[cls_id]

                    extra_groundings.add_detection(class_label, gt_insts=np.unique(
                        labels[(labels[:, 0] == cls_id), 1]).tolist())
                    
                    instance_ids = np.unique(labels[:,1])
                    gt_instances = np.unique(
                        labels[(labels[:, 0] == cls_id), 1])
                    if not np.all(np.isin(gt_instances, instance_ids)):
                        assert False,"gt_instances not in instance_ids"
                # negative labels:
                for cls_id in negative_cls_id_sets:
                    if not (0 <= cls_id < len(text_class_labels)):
                        continue  # 255 / -1 ignore
                    if cls_id in self.filter_out_classes:
                        continue  # continue rather concat
                    class_label = text_class_labels[cls_id]

                    extra_groundings.add_detection(class_label, gt_insts=[])

        if self.lang_query:
            if self.mode == 'train':
                positive_lang_query = min(int(self.lang_query * self.positive_lang_query_ratio), len(
                    self.multi_lang_dict[scene_id]) if scene_id in self.multi_lang_dict else 0)
                negative_lang_query = min(self.lang_query - positive_lang_query, int(
                    positive_lang_query * (1-self.positive_lang_query_ratio)))
            else:
                positive_lang_query = len(
                    self.multi_lang_dict[scene_id]) if scene_id in self.multi_lang_dict else 0
                negative_lang_query = 0  # avoid empty list

            pos_idx = []
            if scene_id in self.multi_lang_dict:  # if there are caption for scene_id
                pos_idx = np.arange(len(self.multi_lang_dict[scene_id]))
            if len(pos_idx) > 0 and self.mode == 'train':
                pos_idx = np.random.choice(
                    pos_idx, positive_lang_query, replace=False)
            for select_idx in pos_idx:
                assert 'description' in self.multi_lang_dict[scene_id][select_idx]

                # filter out some ignore classes like wall, floor
                if self.multi_lang_dict[scene_id][select_idx]['lang_type'].split(':')[0] != 'groundedscenecaption':
                    # groundedscenecaption has filtered before
                    filter_out_flag = False
                    # all other sentence-level uses the same instances ids  
                    for inst_id in self.multi_lang_dict[scene_id][select_idx]['object_ids'][0]:
                        if labels[labels[:, 1] == inst_id, 0][0] in self.filter_out_classes:
                            filter_out_flag = True
                            break
                        if labels[labels[:, 1] == inst_id][0, 0] == self.ignore_label:
                            filter_out_flag = True
                            break
                    if filter_out_flag:
                        continue

                extra_groundings.add_grounding(
                    grounding_text=self.multi_lang_dict[scene_id][select_idx]['description'],
                    gt_insts=self.multi_lang_dict[scene_id][select_idx]['object_ids'],
                    positives=self.multi_lang_dict[scene_id][select_idx]['all_phrases_positions'],
                    grounding_type=self.multi_lang_dict[scene_id][select_idx]['lang_type']
                )

            # random sample negatives from left
            if negative_lang_query > 0 and len(self.multi_lang_source) > 0:
                neg_idx = []
                for select_idx in range(len(self.multi_lang_source)):
                    if self.multi_lang_source[select_idx]['scene_id'] == scene_id:
                        continue
                    if 'description' not in self.multi_lang_source[select_idx]:
                        continue
                    neg_idx.append(select_idx)
                neg_idx = np.asarray(neg_idx)
                neg_idx = np.random.choice(neg_idx, min(
                    negative_lang_query, len(neg_idx)), replace=False)
                for select_idx in neg_idx:
                    extra_groundings.add_grounding(
                        grounding_text=self.multi_lang_source[select_idx]['description'],
                        gt_insts=[
                            []] * len(self.multi_lang_source[select_idx]['all_phrases_positions']),
                        positives=self.multi_lang_source[select_idx]['all_phrases_positions'],
                        grounding_type=self.multi_lang_source[select_idx]['lang_type'],
                    )

            if self.mode == 'train':
                try:
                    extra_groundings.shuffle_grounding()
                except Exception as e:
                    print(e)
                    print("Error in shuffle grounding")
                    pass

        if self.num_concat_texts > 0:
            extra_groundings.concat_multi_grounding(
                tokenizer=self.tokenizer, max_batch_tokens=self.lang_max_token_length, max_tokens=min(
                    512, self.lang_max_token_length),
                num_concat_texts=self.num_concat_texts if self.mode == 'train' else 48,
            )

            if self.mode != 'train':
                if len(extra_groundings.concat_types) < len(extra_groundings.types):
                    print(
                        f'Some langauges are missing as the language clip (16 x 256) during eval: raw has {len(extra_groundings.types)} but get {len(extra_groundings.concat_types)}')

        # scene QA
        instruction_lang_info = []
        if self.lang_query and scene_id in self.instruction_lang_dict and ('scanqa' in self.lang_data_conf or 'taskscheduling' in self.lang_data_conf or 'embodiedplan' in self.lang_data_conf or
                                                                           'objdesc' in self.lang_data_conf or 'scenedesc' in self.lang_data_conf or 'scan2cap' in self.lang_data_conf):
            if self.mode == 'train':
                from utils.sample_utils import sample_by_type

                lang_type_with_index = np.asarray([(d['lang_type'].split(':')[0], i) for i, d in enumerate(
                    self.instruction_lang_dict[scene_id])], dtype=object)
                sampled_lang_type_with_index = sample_by_type(
                    lang_type_with_index, self.max_sample_lang_type_count)
                sampled_index = sampled_lang_type_with_index[:, 1]
            else: # 测试时会全选
                sampled_index = range(
                    len(self.instruction_lang_dict[scene_id]))

            for select_idx in sampled_index:
                instruction_item = self.instruction_lang_dict[scene_id][select_idx]        
                instruction_lang_info.append(
                    lang_info_data.from_instruction_following(
                        instruction_item,
                        train_mode=(self.mode == 'train')
                    ))

        if self.mode == 'train' and self.max_sample_lang_type_count.get("alpaca", 0):
            alpaca_data_sampled = sample(
                self.alpaca_source, self.max_sample_lang_type_count.get("alpaca", 0))
            for instruction_item in alpaca_data_sampled:
                instruction_item['lang_type'] = 'alpaca'
                instruction_lang_info.append(lang_info_data.from_instruction_following(
                    instruction_item,
                ))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
        
        file_name = self.sceneverse_dataset_name + "_" + scene_id
        return [
            coordinates,
            features,
            labels,
            file_name,
            raw_color,
            raw_normals,
            raw_coordinates,
            idx,
            extra_groundings,
            instruction_lang_info
        ]

    def _load_scan(self, scan_id, filter_bkg=False):
        pcd_path = os.path.join(self.data_dir[0], 'scan_data', 'pcd_with_global_alignment', f'{scan_id}.pth')
        inst2label_path = os.path.join(self.data_dir[0], 'scan_data', 'instance_id_to_label', f'{scan_id}.pth')
        superpoint_path = os.path.join(self.data_dir[0], 'scan_data', 'superpoints', f'{scan_id}.pth')
    
        if not os.path.exists(pcd_path):
            return None
        
        assert os.path.exists(pcd_path) and os.path.exists(superpoint_path) and os.path.exists(inst2label_path), f"Missing files for scan {scan_id} in {self.sceneverse_dataset_name} dataset."
            
        pcd_data = torch.load(pcd_path)
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

        superpoints = torch.load(superpoint_path) 
        inst_to_label = torch.load(inst2label_path)
        semantic_label = np.full_like(instance_labels, 0)
        for inst_id in inst_to_label.keys():
            if inst_to_label[inst_id] in self.cat2int.keys():
                mask = instance_labels == inst_id
                if np.sum(mask) == 0:
                    continue
                label = self.cat2int[inst_to_label[inst_id]] 
                scannet_raw_id = self.label_converter.id_to_scannet_raw_id[label]
                semantic_label[mask] = scannet_raw_id
        labels = np.stack([semantic_label, instance_labels], axis=1)   
        scan_data = {}
        scan_data['scene_pcds'] = pcd_data
        scan_data['scene_superpoints'] = superpoints 
        scan_data["labels"] = labels

        return scan_data

    def _convert_pc_to_box(self,obj_pc):
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
        box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
        return center, box_size

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            if label not in self.color_map:
                print(
                    f'WARNING: Found label {label}, temperally changed it to 255')
                label = 255
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    # in ScanNet-200, label = label - 1:  0->255, 1->0, 2->1, 3->2
    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    # in ScanNet-200, label = label + 1: 0->1, 1->2, 2->3
    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

class SceneVerse(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name="scannet", 
        base_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet", 
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        task="instance_segmentation",
        filter_out_classes=[],
        label_offset=0,
        is_elastic_distortion=True,
        lang_query=False,
        positive_lang_query_ratio=0.5,
        lang_max_token_length=256,
        num_concat_texts=4,
        bert_path=BERT_PATH,
        lang_data_conf='',
        sample_class_labels=False,
        axis_align_coord=False,
        datasets = ['ScanNet', '3RScan', 'MultiScan', "ARKitScenes", 'HM3D'], 
        dataset_sample_ratio = 1.0,
        debug = False,
    ):
        self.debug = debug
        self.base_dir = base_dir
        self.label_offset = label_offset
        self.dataset_name = dataset_name
        self.dataset_sample_ratio = dataset_sample_ratio
        self.mode = mode
        all_datasets = []
        for dataset in datasets:
            assert dataset in ['ScanNet', '3RScan', 'MultiScan', "ARKitScenes", 'HM3D'], f"dataset {dataset} not in ['ScanNet', '3RScan', 'MultiScan', 'ARKitScenes', 'HM3D']"
            data = SceneVerseBase(
                dataset_name=dataset_name,
                data_dir=os.path.join(base_dir, dataset),
                label_db_filepath=label_db_filepath,
                color_mean_std=color_mean_std,
                mode=mode,
                add_colors=add_colors,
                add_normals=add_normals,
                add_raw_coordinates=add_raw_coordinates,
                num_labels=num_labels,
                ignore_label=ignore_label,
                volume_augmentations_path=volume_augmentations_path,
                image_augmentations_path=image_augmentations_path,
                task=task,
                filter_out_classes=filter_out_classes,
                label_offset=label_offset,
                is_elastic_distortion=is_elastic_distortion,
                lang_query=lang_query,
                positive_lang_query_ratio=positive_lang_query_ratio,
                lang_max_token_length=lang_max_token_length,
                num_concat_texts=num_concat_texts,
                bert_path=BERT_PATH,
                lang_data_conf=lang_data_conf,
                sample_class_labels=sample_class_labels,
                axis_align_coord=axis_align_coord,
                scenes=self._get_scenes(dataset, mode, dataset_sample_ratio) 
            )
            all_datasets.append(data)
        
        self.dataset = ConcatDataset(all_datasets) if all_datasets else None
        self._print_info()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]

    @property
    def label_info(self):
        if isinstance(self.dataset, ConcatDataset):
            return self.dataset.datasets[0].label_info
        else:
            return self.dataset.label_info

    def _remap_model_output(self, output):
        if isinstance(self.dataset, ConcatDataset):
            return self.dataset.datasets[0]._remap_model_output(output)
        else:
            return self.dataset._remap_model_output(output)


    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            if label not in self.color_map:
                print(
                    f'WARNING: Found label {label}, temperally changed it to 255')
                label = 255
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)
    
      
    def _get_scenes(self, dataset, mode, ratio=1.0):
        if mode == "validation":
            mode = "test"
        assert mode in ["train", "test"]
        if mode == "train":
            splits_path = os.path.join(self.base_dir, "splits", dataset+f"_{mode}_scans.txt") 
        elif mode == "test":
            splits_path = os.path.join(self.base_dir, "splits", dataset+f"_{mode}_scans.txt") 
        with open(splits_path, "r") as f:
            scenes = f.read().splitlines()
        num_scenes = len(scenes)
        ratio = max(0.0, min(ratio, 1.0)) 
        selected_count = max(1, int(num_scenes * ratio)) if num_scenes > 0 else 0
        scenes = scenes[:selected_count]
        return scenes
    
    def _print_info(self):
        print("="*70)
        print(f"ORS3D-60K dataset Scene num [Total] ({self.mode} mode): {len(self)}")
        print("="*70)

class LabelConverter(object):
    def __init__(self, file_path):
        self.raw_name_to_id = {}
        self.nyu40id_to_id = {}
        self.nyu40_name_to_id = {}
        self.scannet_name_to_scannet_id = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4,
            'door':5, 'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.id_to_scannetid = {}
        self.scannet_raw_id_to_raw_name = {}
        self.id_to_scannet_raw_id = {}

        with open(file_path, encoding='utf-8') as fd:
            rd = list(csv.reader(fd, delimiter="\t", quotechar='"'))
            for i in range(1, len(rd)):
                raw_id = i - 1
                scannet_raw_id = int(rd[i][0])
                self.id_to_scannet_raw_id[raw_id] = scannet_raw_id
                raw_name = rd[i][1]
                nyu40_id = int(rd[i][4])
                nyu40_name = rd[i][7]
                self.raw_name_to_id[raw_name] = raw_id
                self.scannet_raw_id_to_raw_name[scannet_raw_id] = raw_name
                self.nyu40id_to_id[nyu40_id] = raw_id
                self.nyu40_name_to_id[nyu40_name] = raw_id
                if nyu40_name not in self.scannet_name_to_scannet_id:
                    self.id_to_scannetid[raw_id] = self.scannet_name_to_scannet_id['others']
                else:
                    self.id_to_scannetid[raw_id] = self.scannet_name_to_scannet_id[nyu40_name]

        self.orgInstID_to_id = {id : id - 1 for id in range(1, 257)}
        self.orgInstID_to_id[0] = -100
        self.scannet_raw_id_to_scannet200_id = {}
        self.scannet200_id_to_scannet_raw_id = {}
        for v, k in enumerate(VALID_CLASS_IDS_200):
            self.scannet_raw_id_to_scannet200_id[k] = v
            self.scannet200_id_to_scannet_raw_id[v] = k


