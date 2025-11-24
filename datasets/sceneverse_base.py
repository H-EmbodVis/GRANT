import os
import copy
import json
import jsonlines
import random

from tqdm import tqdm
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset

from ..data_utils import LabelConverter, build_rotate_mat
from ..data_utils import convert_pc_to_box, construct_bbox_corners, \
                            merge_tokens, eval_ref_one_sample, is_explicitly_view_dependent
from .data_augmentor import DataAugmentor
from .constant import CLASS_LABELS_200


class ScanBase(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.pc_type = cfg.data.args.pc_type
        self.max_obj_len = cfg.data.args.max_obj_len
        self.num_points = cfg.data.args.num_points
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        self.debug = cfg.debug.flag
        self.debug_size = cfg.debug.debug_size
        self.subset_ratio = getattr(cfg.data.args, 'subset_ratio', 0)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)
        self.scannet_dir = cfg.data.scan_family_base

        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        self.int2cat = json.load(open(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

        self.use_scene_cap = getattr(cfg.data.args, 'use_scene_cap', False)

    def _load_split(self, split):
        if 'scannet' in self.__class__.__name__.lower():
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + ".txt")
        else:
            split_file = os.path.join(self.base_dir, 'annotations/splits/'+ split + "_split.txt")

        scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
        scan_ids = sorted(scan_ids)

        return scan_ids

    def _load_scan(self, scan_ids, filter_bkg=False):
        scans = {}
        for scan_id in tqdm(scan_ids):
            pcd_path = os.path.join(self.base_dir, 'scan_data', 'pcd_with_global_alignment', f'{scan_id}.pth')
            inst2label_path = os.path.join(self.base_dir, 'scan_data', 'instance_id_to_label', f'{scan_id}.pth')

            if not os.path.exists(pcd_path):
                continue
            pcd_data = torch.load(pcd_path)
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # build obj_pcds
            inst_to_label = torch.load(inst2label_path)
            obj_pcds = []
            inst_ids = []
            inst_labels = []
            bg_indices = np.full((points.shape[0], ), 1, dtype=np.bool_)
            for inst_id in inst_to_label.keys():
                if inst_to_label[inst_id] in self.cat2int.keys():
                    mask = instance_labels == inst_id
                    if np.sum(mask) == 0:
                        continue
                    obj_pcds.append(pcds[mask])
                    inst_ids.append(inst_id)
                    inst_labels.append(self.cat2int[inst_to_label[inst_id]])
                    if inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask] = False
            if filter_bkg:
                selected_obj_idxs = [i for i, obj_label in enumerate(inst_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
                if len(selected_obj_idxs) == 0:
                    continue
            scans[scan_id] = {}
            scans[scan_id]['obj_pcds'] = obj_pcds
            scans[scan_id]['inst_labels'] = inst_labels
            scans[scan_id]['inst_ids'] = inst_ids
            scans[scan_id]['bg_pcds'] = pcds[bg_indices]
            obj_center = []
            obj_box_size = []
            for obj_pcd in obj_pcds:
                _c, _b = convert_pc_to_box(obj_pcd)
                obj_center.append(_c)
                obj_box_size.append(_b)
            scans[scan_id]['obj_center'] = obj_center
            scans[scan_id]['obj_box_size'] = obj_box_size

            obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
            if os.path.exists(obj_mask_path):
                obj_label_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
                obj_labels = np.load(obj_label_path)[:50]
                obj_l = []
                bg_indices = np.full((pcds.shape[0], ), 1, dtype=np.bool_)
                for i in range(obj_mask.shape[0]):
                    mask = obj_mask[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                        obj_l.append(obj_labels[i])
                        bg_indices[mask == 1] = False
                scans[scan_id]['obj_pcds_pred'] = obj_pcds
                scans[scan_id]['inst_labels_pred'] = obj_l
                scans[scan_id]['bg_pcds_pred'] = pcds[bg_indices]
                obj_center_pred = []
                obj_box_size_pred = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center_pred.append(_c)
                    obj_box_size_pred.append(_b)
                scans[scan_id]['obj_center_pred'] = obj_center_pred
                scans[scan_id]['obj_box_size_pred'] = obj_box_size_pred
        return scans

    def _load_lang(self, cfg, scan_ids):
        caption_source = cfg.sources
        json_data = []
        lang_data = []
        valid_scan_ids = []

        if self.use_scene_cap:
            scene_cap_file = os.path.join(self.base_dir, 'annotations/scene_cap.json')
            if not os.path.exists(scene_cap_file):
                self.scene_caps = {}
            else:
                with open(scene_cap_file, 'r') as f:
                    self.scene_caps = json.load(f)
        else:
            self.scene_caps = None

        for anno_type in caption_source:
            if anno_type == 'anno':
                anno_file = os.path.join(self.base_dir, 'annotations/anno.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
            elif anno_type == 'referit3d':
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if len(item['tokens']) <= 24:
                                json_data.append(item)
                if cfg.referit3d.sr3d_plus_aug:
                    anno_file = os.path.join(self.base_dir, 'annotations/refer/sr3d+.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if len(item['tokens']) <= 24:
                                json_data.append(item)
            elif anno_type == 'scanrefer':
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        json_data.append(item)
            elif anno_type == 'sgrefer':
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_ref_{anno_type}.json')
                    json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
            elif anno_type == 'sgcaption':
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_obj_caption_{anno_type}.json')
                    json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
            else:
                if 'obj_caption' in anno_type:
                    anno_file = os.path.join(self.base_dir, f'annotations/ssg_{anno_type}.json')
                else:
                    anno_file = os.path.join(self.base_dir, f'annotations/ssg_ref_{anno_type}.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
        for item in json_data:
            if item['scan_id'] in scan_ids and item['instance_type'] not in ['wall', 'floor', 'ceiling']:
                lang_data.append(item)
                if item['scan_id'] not in valid_scan_ids:
                    valid_scan_ids.append(item['scan_id'])
        valid_scan_ids = sorted(valid_scan_ids)
        if self.subset_ratio > 0:
            valid_scan_ids = valid_scan_ids[:int(self.subset_ratio*len(valid_scan_ids))]
            lang_data = [item for item in lang_data if item['scan_id'] in valid_scan_ids]

        if self.debug and self.debug_size != -1:
            valid_scan_ids = valid_scan_ids[:self.debug_size]
            lang_data = [item for item in lang_data if item['scan_id'] in valid_scan_ids]

        return lang_data, valid_scan_ids

    def _getitem_pretrain(self, index, is_rscan=False):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        if is_rscan and hasattr(item, 'sentence'):
            sentence = item['sentence']
        else:
            sentence = item['utterance']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            remained_obj_idx = [i for i in range(len(obj_pcds))]
            random.shuffle(remained_obj_idx)
            selected_obj_idxs = remained_obj_idx[:self.max_obj_len]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len

        if not self.aug_cfg:
            obj_fts, obj_locs, _, obj_labels = self._obj_processing_post(obj_pcds, obj_labels,
                                                                         is_need_bbox=True,
                                                                         rot_aug=self.rot_aug)
        else:
            obj_fts, obj_locs, _, obj_labels = self._obj_processing_aug(obj_pcds, obj_labels,
                                                                        is_need_bbox=True)

        data_dict = {'scan_id': scan_id,
                     'sentence': sentence,
                     'obj_fts': obj_fts,
                     'obj_locs': obj_locs,
                     'obj_labels': obj_labels} 

        return data_dict

    def _getitem_obj_pretrain(self, index):
        scan_id = self.scan_ids[index]
        sentence = 'placeholder'

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                             if (self.int2cat[obj_label] in CLASS_LABELS_200) and (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            remained_obj_idx = [i for i in range(len(obj_pcds))]
            random.shuffle(remained_obj_idx)
            selected_obj_idxs = remained_obj_idx[:self.max_obj_len]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len

        if not self.load_scene_pcds:
            if not self.aug_cfg:
                obj_fts, obj_locs, _, obj_labels = self._obj_processing_post(obj_pcds,
                                                                             obj_labels,
                                                                             is_need_bbox=True,
                                                                             rot_aug=self.rot_aug)
            else:
                obj_fts, obj_locs, _, obj_labels = self._obj_processing_aug(obj_pcds,
                                                                            obj_labels,
                                                                            is_need_bbox=True)
        else:
            assert self.aug_cfg
            bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, _, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds,
                                                                                             bg_pcds,
                                                                                             obj_labels,
                                                                                             is_need_bbox=True)

        if not self.load_scene_pcds:
            data_dict = {'scan_id': scan_id,
                        'sentence': sentence,
                        'obj_fts': obj_fts,
                        'obj_locs': obj_locs,
                        'obj_labels': obj_labels}
        else:
            data_dict = {'scan_id': scan_id,
                         'sentence': sentence, 
                         'obj_locs': obj_locs, 
                         'obj_labels': obj_labels, 
                         'obj_pcds_masks': obj_pcds_masks, 
                         'scene_pcds': scene_pcds}
        return data_dict

    def _getitem_refer(self, index):
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_instance = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['utterance'].split(' '))

        if self.use_scene_cap:
            scene_caps = self.scene_caps.get(scan_id)
            if scene_caps is not None:
                scene_caps = scene_caps['captions']
                scene_cap = scene_caps[np.random.choice(len(scene_caps))]
            else:
                scene_cap = "This is a scene."

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_ids = self.scan_data[scan_id]['inst_ids'] # N
            assert tgt_object_instance in obj_ids, str(tgt_object_instance) + ' not in ' + str(obj_ids) + '-' + scan_id
            tgt_object_id = obj_ids.index(tgt_object_instance)
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
        elif self.pc_type == 'pred':
            obj_ids = self.scan_data[scan_id]['inst_ids'] # N
            tgt_object_id = obj_ids.index(tgt_object_instance)
            gt_pcd = self.scan_data[scan_id]["obj_pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        if not self.load_scene_pcds:
            if not self.aug_cfg:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds,
                                                                                     obj_labels,
                                                                                     is_need_bbox=True,
                                                                                     rot_aug=self.rot_aug)
            else:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_aug(obj_pcds,
                                                                                    obj_labels,
                                                                                    is_need_bbox=True)
        else:
            assert self.aug_cfg
            if self.pc_type == 'pred':
                bg_pcds = self.scan_data[scan_id]['bg_pcds_pred']
            else:
                bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds,
                                                                                                     bg_pcds,
                                                                                                     obj_labels,
                                                                                                     is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_pcds) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_pcds) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count_multi'][self.label_converter.id_to_scannetid
                                                                  [tgt_object_label]] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]),  
            "tgt_object_label": torch.LongTensor([tgt_object_label]),  
            "obj_locs": obj_locs,  # N, 3
            "obj_labels": obj_labels,  # N,
            "obj_boxes": obj_boxes,  # N, 6
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50,
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        }
        if self.load_scene_pcds:
            data_dict["scene_pcds"] = scene_pcds
            data_dict["obj_pcds_masks"] = obj_pcds_masks
        else:
            data_dict["obj_fts"] = obj_fts  # N, 6

        if self.use_scene_cap:
            data_dict["scene_cap"] = scene_cap
        return data_dict

    def _getitem_perscene(self, index):
        item = self.lang_data[index]
        scan_id =  item[0]['scan_id']
        list_item_id = [_i['item_id'] for _i in item]
        list_tgt_object_instance = [int(_i['target_id']) for _i in item]
        list_tgt_object_name = [_i['instance_type'] for _i in item]
        list_sentence = [_i['utterance'][0] for _i in item]
        list_token = [_i['utterance'][1] for _i in item]
        list_mask = [_i['utterance'][2] for _i in item]
        list_is_view_dependent = [is_explicitly_view_dependent(sentence.split(' ')) for sentence in list_sentence]

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_ids = self.scan_data[scan_id]['inst_ids'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        list_tgt_object_id = [obj_ids.index(_ins) for _ins in list_tgt_object_instance]

        if self.pc_type == 'gt':
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                            if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        else:
            selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        list_tgt_object_label = []
        list_tgt_object_id_iou25_list = []
        list_tgt_object_id_iou50_list = []
        list_is_multiple = []
        list_is_hard = []
        for idx, _ in enumerate(list_item_id):
            item_id = list_item_id[idx]
            tgt_object_id = list_tgt_object_id[idx]
            tgt_object_name = list_tgt_object_name[idx]

            if self.pc_type == 'gt':
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
                tgt_object_label = obj_labels[tgt_object_id]
                tgt_object_id_iou25_list = [tgt_object_id]
                tgt_object_id_iou50_list = [tgt_object_id]
                assert self.int2cat[tgt_object_label] == tgt_object_name, str(self.int2cat[tgt_object_label]) + '-' + tgt_object_name
            elif self.pc_type == 'pred':
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][tgt_object_id]
                gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
                tgt_object_id = -1
                tgt_object_id_iou25_list = []
                tgt_object_id_iou50_list = []
                tgt_object_label = self.cat2int[tgt_object_name]
                # find tgt iou 25
                for i, _ in enumerate(obj_pcds):
                    obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                    if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                        construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                        tgt_object_id = i
                        tgt_object_id_iou25_list.append(i)
                # find tgt iou 50
                for i, _ in enumerate(obj_pcds):
                    obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                    if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                        construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                        tgt_object_id_iou50_list.append(i)
            # build unique multiple
            is_multiple = self.scan_data[scan_id]['label_count'][self.label_converter.id_to_scannetid
                                                                    [tgt_object_label]] > 1
            is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

            list_tgt_object_id[idx] = tgt_object_id
            list_tgt_object_label.append(tgt_object_label)
            list_tgt_object_id_iou25_list.append(tgt_object_id_iou25_list)
            list_tgt_object_id_iou50_list.append(tgt_object_id_iou50_list)
            list_is_multiple.append(is_multiple)
            list_is_hard.append(is_hard)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            selected_obj_idxs = [x for x in list_tgt_object_id if x != -1]
            for idx, _ in enumerate(list_tgt_object_id):
                selected_obj_idxs.extend(list_tgt_object_id_iou25_list[idx])
                selected_obj_idxs.extend(list_tgt_object_id_iou50_list[idx])
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]

            list_tgt_object_id_tmp = []
            for tgt_object_id in list_tgt_object_id:
                list_tgt_object_id_tmp.append(selected_obj_idxs.index(tgt_object_id)  if tgt_object_id != -1 else -1)
            list_tgt_object_id = list_tgt_object_id_tmp
            list_tgt_object_id_iou25_list_tmp = []
            for tgt_object_id_iou25_list in list_tgt_object_id_iou25_list:
                list_tgt_object_id_iou25_list_tmp.append([selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list])
            list_tgt_object_id_iou25_list = list_tgt_object_id_iou25_list_tmp
            list_tgt_object_id_iou50_list_tmp = []
            for tgt_object_id_iou50_list in list_tgt_object_id_iou50_list:
                list_tgt_object_id_iou50_list_tmp.append([selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list])
            list_tgt_object_id_iou50_list = list_tgt_object_id_iou50_list_tmp
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        for idx, _id in enumerate(list_tgt_object_id):
            if _id == -1:
                list_tgt_object_id[idx] = len(obj_pcds)

        # build scene
        assert self.aug_cfg
        if self.pc_type == 'pred':
            bg_pcds = self.scan_data[scan_id]['bg_pcds_pred']
        else:
            bg_pcds = self.scan_data[scan_id]['bg_pcds']
        obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds,
                                                                                                 bg_pcds,
                                                                                                 obj_labels,
                                                                                                 is_need_bbox=True)

        # build iou25 and iou50
        list_tgt_object_id_iou25 = torch.zeros((len(list_sentence), len(obj_pcds) + 1)).long()
        list_tgt_object_id_iou50 = torch.zeros((len(list_sentence), len(obj_pcds) + 1)).long()
        for _rid, tgt_id in enumerate(list_tgt_object_id_iou25_list):
            for _cid in tgt_id:
                list_tgt_object_id_iou25[_rid, _cid] = 1
        for _rid, tgt_id in enumerate(list_tgt_object_id_iou50_list):
            for _cid in tgt_id:
                list_tgt_object_id_iou50[_rid, _cid] = 1

        data_dict = {
            "sentence": list_sentence,  # list, len L
            "txt_ids": list_token, # Tensor, L
            "txt_masks": torch.LongTensor(list_mask), # Tensor, L
            "tgt_object_id": torch.LongTensor(list_tgt_object_id), # Tensor, L
            "tgt_object_label": torch.LongTensor(list_tgt_object_label), # Tensor, L
            "scene_pcds": scene_pcds, # N_pts, 6
            "obj_locs": obj_locs, # Tensor N, 6
            "obj_labels": obj_labels, # Tensor, N
            "obj_boxes": obj_boxes, # Tensor, N, 6
            "data_idx": item_id, # str, 1
            "tgt_object_id_iou25": list_tgt_object_id_iou25, # Tensor, (L, N+1)
            "tgt_object_id_iou50": list_tgt_object_id_iou50, # Tensor, (L, N+1)
            "is_multiple": torch.LongTensor(list_is_multiple), # list, L
            "is_view_dependent": torch.LongTensor(list_is_view_dependent), # List, L
            "is_hard": torch.LongTensor(list_is_hard), # List, L
            "obj_pcds_masks": obj_pcds_masks # Tensor, (N, 1024)
        }
        return data_dict

    def _obj_processing_post(self, obj_pcds, obj_labels, is_need_bbox=False, rot_aug=False):
        # rotate obj
        rot_matrix = build_rotate_mat(self.split, rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels

    def _obj_processing_aug(self, obj_pcds, obj_labels, is_need_bbox=False):
        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        if isinstance(obj_pcds, list):
            obj_pcds = torch.Tensor(np.array(obj_pcds))
        obj_sizes = torch.Tensor(np.array(data_dict['obj_sizes']))

        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = torch.Tensor(obj_sizes)
        # size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])

        # convert to torch
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]

        return obj_pcds, obj_locs, obj_boxes, obj_labels

    def _scene_processing_aug(self, obj_pcds, bg_pcds, obj_labels, is_need_bbox=False):
        obj_len = len(obj_pcds)
        # sample background points
        fg_points_num = len(obj_pcds) * self.num_points
        assert fg_points_num < self.max_pcd_num_points
        bg_points_num = min(self.max_pcd_num_points - fg_points_num, self.bg_points_num)
        assert len(bg_pcds) > 0
        assert bg_points_num > 0
        bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        bg_pcds = bg_pcds[bg_points_indices]

        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'bg_pcds': torch.Tensor(bg_pcds), 
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        if isinstance(obj_pcds, list):
            obj_pcds = torch.Tensor(np.array(obj_pcds))
        obj_sizes = torch.Tensor(np.array(data_dict['obj_sizes']))
        bg_pcds = data_dict['bg_pcds']
        assert len(obj_pcds) * obj_pcds[0].shape[0] == fg_points_num

        scene_pcds = np.vstack([np.array(obj_pcds.reshape(-1, 6)), np.array(bg_pcds)])

        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = torch.Tensor(obj_sizes)
        # size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])

        # generate obj point indices masks
        obj_pcds_masks = []
        offset = 0
        for _j in range(obj_len):
            mask = np.arange(self.num_points) + offset
            assert len(mask) == len(obj_pcds[_j])
            obj_pcds_masks.append(mask)
            offset += self.num_points

        # convert to torch
        obj_labels = torch.LongTensor(obj_labels)
        obj_pcds_masks = torch.from_numpy(np.array(obj_pcds_masks))

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_pcds_masks.shape[0] == obj_locs.shape[0]

        return obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds

    def _getitem_finalrefer(self, index):
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id = item['scan_id']
        tgt_object_instance = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['utterance'].split(' '))

        txt_ids = item['txt_ids']
        txt_masks = item['txt_masks']
        if self.use_scene_cap:
            scene_caps = self.scene_caps.get(scan_id)
            if scene_caps is not None:
                scene_cap = copy.deepcopy(scene_caps[np.random.choice(len(scene_caps))])
            else:
                scene_cap = copy.deepcopy(self.default_scene_cap)
        if self.use_scene_cap:
            scene_txt_ids = scene_cap['scene_txt_ids']
            scene_txt_masks = scene_cap["scene_txt_masks"]
            scene_txt_ids, scene_txt_masks = merge_tokens(
                scene_txt_ids, scene_txt_masks, txt_ids, txt_masks,
                max_len=self.max_scene_cap_len, tokenizer=self.tokenizer
            )

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds']  # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels']  # N
            obj_ids = self.scan_data[scan_id]['inst_ids']  # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            obj_ids = self.scan_data[scan_id]['inst_ids_pred']  # N

        assert tgt_object_instance in obj_ids, str(tgt_object_instance) + ' not in ' + str(obj_ids) + '-' + scan_id
        tgt_object_id = obj_ids.index(tgt_object_instance)
        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                     if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                     and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                     if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["obj_pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        if not self.load_scene_pcds:
            if not self.aug_cfg:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds,
                                                                                     obj_labels,
                                                                                     is_need_bbox=True,
                                                                                     rot_aug=self.rot_aug)
            else:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_aug(obj_pcds,
                                                                                    obj_labels,
                                                                                    is_need_bbox=True)
        else:
            assert self.aug_cfg
            if self.pc_type == 'pred':
                bg_pcds = self.scan_data[scan_id]['bg_pcds_pred']
            else:
                bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds,
                                                                                                     bg_pcds,
                                                                                                     obj_labels,
                                                                                                     is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_pcds) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_pcds) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "txt_ids": torch.LongTensor(txt_ids),
            "txt_masks": torch.LongTensor(txt_masks),
            "tgt_object_id": torch.LongTensor([tgt_object_id]),  # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]),  # 1
            "obj_locs": obj_locs,  # N, 3
            "obj_labels": obj_labels,  # N,
            "obj_boxes": obj_boxes,  # N, 6
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50,
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        }
        if self.load_scene_pcds:
            data_dict["scene_pcds"] = scene_pcds
            data_dict["obj_pcds_masks"] = obj_pcds_masks
        else:
            data_dict["obj_fts"] = obj_fts  # N, 6

        if self.use_scene_cap:
            data_dict["scene_cap"] = scene_cap["scene_cap"]
            data_dict["scene_txt_ids"] = torch.LongTensor(scene_txt_ids)
            data_dict["scene_txt_masks"] = torch.LongTensor(scene_txt_masks)
        return data_dict
