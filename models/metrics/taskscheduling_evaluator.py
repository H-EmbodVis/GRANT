import argparse
import json
from collections import defaultdict
import re
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt

import re
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from benchmark.evaluate_semantic_instance import evaluate as mask3d_det_evaluation
import glob
import pickle

try:
    from models.metrics.multi3drefer_evaluator import parse_prediction,Multi3DReferEvaluator
except:
    from multi3drefer_evaluator import parse_prediction,Multi3DReferEvaluator
import copy


def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(batch_boxes_1_bound[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(batch_boxes_1_bound[:, 1], 3, dim=1)
    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(batch_boxes_2_bound[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(batch_boxes_2_bound[:, 1], 3, dim=1)
    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)
    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                        torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
    return iou.flatten()


def get_target_subtask_id(gt_subtasks, phrase_in_pred_action, output_line, gt_actions):
    assert phrase_in_pred_action != ""
    phrase_in_pred_action = phrase_in_pred_action.strip()
    target_subtask_idx = None
    for subtask_index, subtask in enumerate(gt_subtasks):
        if phrase_in_pred_action.lower() in subtask['subtask_description'].lower():
            target_subtask_idx = subtask_index
            break
        elif subtask['label'].lower() in phrase_in_pred_action.lower():
            target_subtask_idx = subtask_index
            break
        elif phrase_in_pred_action.lower() in subtask['label'].lower():
            target_subtask_idx = subtask_index
            break
        elif "tv" in subtask['label'] or "TV" in subtask['label']:
            if "television" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "printer" in subtask['label']:
            if "monitor" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "water heater" in subtask['label']:
            if "waterheater" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "picture" in subtask['label']:
            if "painting" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "painting" in subtask['label']:
            if "picture" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "microwave" in subtask['label']:
            if "oven" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        elif "printer" in subtask['label']:
            if "machine" in phrase_in_pred_action.lower():
                target_subtask_idx = subtask_index
                break
        else:
            pred_words = set(phrase_in_pred_action.lower().split())
            gt_words = set(subtask['label'].lower().split())
            if pred_words & gt_words:
                target_subtask_idx = subtask_index
                break

    if target_subtask_idx == None:
        target_sentences = [subtask['subtask_description'].lower() for subtask in gt_subtasks]
        best_sentence, max_overlap = find_most_similar_sentence(phrase_in_pred_action.lower(), target_sentences)
        if max_overlap > 2:
            target_subtask_idx = target_sentences.index(best_sentence)

    gt_phrase_list = []
    for action in gt_actions:
        match = re.search(r"<p>(.*?)</p>", action['action'].lower())
        if match:
            gt_phrase = match.group(1).strip()
        gt_phrase = re.sub(r'^\\bThe\\b ', '', gt_phrase, flags=re.IGNORECASE)
        gt_phrase_list.append(gt_phrase)
    processed_phrase_in_pred_action = re.sub(r'^\\bThe\\b ', '', phrase_in_pred_action, flags=re.IGNORECASE).lower()
    best_sentence, max_overlap = find_most_similar_sentence(processed_phrase_in_pred_action, gt_phrase_list)
    if max_overlap > 0:
        tgt_index = gt_phrase_list.index(best_sentence)
        target_subtask_idx_full_text = gt_actions[tgt_index]["subtask_index"]-1
        if target_subtask_idx != target_subtask_idx_full_text:
                print(f"input_line is:{output_line}")
                print(f"full_text matched sentence is:{best_sentence}")
                print(f"max_overlap is:{max_overlap}")
                print(f"all gt phrases is:{gt_phrase_list}")
                if target_subtask_idx != None:
                    print(f"Double check: phrase match:{gt_subtasks[target_subtask_idx]['label']} vs full_text match:{gt_subtasks[target_subtask_idx_full_text]['label']}")
                    target_subtask_idx = target_subtask_idx_full_text
                else:
                    print(f"full_text match:{gt_subtasks[target_subtask_idx_full_text]['label']}")
    else:
        if target_subtask_idx != None:
            print(f"input_line is:{output_line}")
            print(f"task match result is:{gt_subtasks[target_subtask_idx]}")
            print(f"full_text matched sentence is:{best_sentence}")
            print(f"max_overlap is:{max_overlap}")
            print(f"all gt phrases is:{gt_phrase_list}")

    return target_subtask_idx


def parse_subtask_info(sentence):
    pattern = r'\(\s*(start\s+subtask|recheck\s+subtask|subtask)\s+(\d+)\s*\)'
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        task_type_raw = match.group(1).lower()
        number = int(match.group(2))

        if "start" in task_type_raw:
            task_type = "start"
        elif "recheck" in task_type_raw:
            task_type = "recheck"
        else:
            task_type = "normal"

        return {
            "subtask_number": number,
            "type": task_type
        }
    return None

def get_target_subtask_id_v2(gt_subtasks, phrase_in_pred_action, output_line, gt_actions):
    subtask_info = parse_subtask_info(output_line)
    target_subtask_idx = subtask_info['subtask_number'] - 1 if subtask_info else None
    target_subtask_type = subtask_info['type'] if subtask_info else None
    return target_subtask_idx


def find_most_similar_sentence(sentence_a, target_sentences):
    words_a = set(sentence_a.split())

    max_overlap = 0
    best_sentence = None

    for sentence in target_sentences:
        words_b = set(sentence.split())
        overlap = len(words_a & words_b)

        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence

    return best_sentence, max_overlap

def get_mask_iou_and_bbox_iou(pred_id, gt_id, gt_query_id, mask):
    pred_top1_mask = mask['pred_inst_masks'][0][:,pred_id].squeeze().bool()
    pred_id = [pred_id]
    gt_id = [gt_id]
    if mask['pred_inst_masks'][0][:,pred_id].shape[-1]!=1:
        pred_mask = torch.any(mask['pred_inst_masks'][0][:,pred_id].squeeze(),dim=1)
    else:
        pred_mask = mask['pred_inst_masks'][0][:,pred_id].squeeze()

    if mask['target_full'][0]['masks'][gt_id,:].shape[0] !=1:
        gt_mask = torch.any(mask['target_full'][0]['masks'][gt_id,:].squeeze(),dim=0)
    else:
        gt_mask = mask['target_full'][0]['masks'][gt_id,:].squeeze()
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = (pred_mask & gt_mask).sum()
    outer = (pred_mask | gt_mask).sum()
    iou = inter / (outer + 1e-8)
    top_1_pred_points = mask["original_coordinates"][0][pred_top1_mask]

    if top_1_pred_points.size:
        min_vals,max_vals  = top_1_pred_points.min(axis=0),top_1_pred_points.max(axis=0)
        pred_bbox = np.vstack((min_vals, max_vals))
        gt_points = mask["original_coordinates"][0][gt_mask]
        min_vals,max_vals  = gt_points.min(axis=0),gt_points.max(axis=0)
        gt_bbox = np.vstack((min_vals, max_vals))
        all_pred_bbox = torch.tensor(np.stack([pred_bbox],axis=0))
        all_gt_bbox = torch.tensor(np.stack([gt_bbox],axis=0))
        top_1_bbox_iou = get_batch_aabb_pair_ious(all_pred_bbox , all_gt_bbox).tolist()[0]
    else:
        top_1_bbox_iou = 0.0

    pred_top1_mask = mask['pred_inst_masks'][0][:,gt_query_id].squeeze().bool()
    gt_query_id = [gt_query_id]
    gt_id = [gt_id]
    if mask['pred_inst_masks'][0][:,gt_query_id].shape[-1]!=1:
        pred_mask = torch.any(mask['pred_inst_masks'][0][:,gt_query_id].squeeze(),dim=1)
    else:
        pred_mask = mask['pred_inst_masks'][0][:,gt_query_id].squeeze()

    if mask['target_full'][0]['masks'][gt_id,:].shape[0] !=1:
        gt_mask = torch.any(mask['target_full'][0]['masks'][gt_id,:].squeeze(),dim=0)
    else:
        gt_mask = mask['target_full'][0]['masks'][gt_id,:].squeeze()
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = (pred_mask & gt_mask).sum()
    outer = (pred_mask | gt_mask).sum()
    encoder_gt_iou = inter / (outer + 1e-8)
    top_1_pred_points = mask["original_coordinates"][0][pred_top1_mask]

    if top_1_pred_points.size:
        min_vals,max_vals  = top_1_pred_points.min(axis=0),top_1_pred_points.max(axis=0)
        pred_bbox = np.vstack((min_vals, max_vals))
        gt_points = mask["original_coordinates"][0][gt_mask]
        min_vals,max_vals  = gt_points.min(axis=0),gt_points.max(axis=0)
        gt_bbox = np.vstack((min_vals, max_vals))
        all_pred_bbox = torch.tensor(np.stack([pred_bbox],axis=0))
        all_gt_bbox = torch.tensor(np.stack([gt_bbox],axis=0))
        encoder_gt_top_1_bbox_iou = get_batch_aabb_pair_ious(all_pred_bbox , all_gt_bbox).tolist()[0]
    else:
        encoder_gt_top_1_bbox_iou = 0.0

    iou_dict = {
        'gt_label': gt_id[0],
        'gt_query_label': gt_query_id[0],
        'mask_iou': iou.item(),
        'bbox_iou': top_1_bbox_iou,
        'pred_id': pred_id[0],

        'encoder_gt_mask_iou': encoder_gt_iou.item(),
        'encoder_gt_bbox_iou': encoder_gt_top_1_bbox_iou,
    }
    return iou_dict

def extract_after_think(s):
    index = s.find('</think>')
    if index != -1:
        return s[index + len('</think>'):]
    else:
        return s


def get_scheduling_info(gt_label, gt_query_label, instance, mask):
    pred_action_to_subtask_idx = []
    pred_results_info = []
    grounding_result = instance["grounding_result"]
    if grounding_result == None:
        return pred_action_to_subtask_idx, pred_results_info
    gt_subtasks = instance["subtasks"]
    gt_actions = instance["actions"]
    output_language = instance["output_language"]

    p_tag_pattern = r"<p>(.*?)</p>"
    ref_tag_pattern = r"<GRU>"

    for idx, action in enumerate(gt_actions):
        action_target_id = action['target_id']
        for subtask in gt_subtasks:
            if subtask['target_id'] == action_target_id:
                subtask['target_label_id'] = gt_label[idx]
                subtask['target_label_query_id'] = gt_query_label[idx]
                break

    id_count = 0
    if "</think>" in output_language:
        index = output_language.find('</think>')
        if "<GRU>" in output_language[:index]:
            ref_count = re.findall(r'<GRU>', output_language[:index])
            id_count += len(ref_count)
        output_language = output_language[index + len('</think>'):]

    output_lines = output_language.split("\n")
    if len(output_lines) < 1:
        return pred_action_to_subtask_idx, pred_results_info

    for i in range(len(output_lines)):
        output_line = output_lines[i]
        if "<GRU>" in output_line:
            ref_count = re.findall(r'<GRU>', output_line)
            target_phrases = re.findall(p_tag_pattern, output_line)

            target_phrase = " ".join(target_phrases) if target_phrases else ""
            target_phrase = re.sub(ref_tag_pattern, '', target_phrase)

            if target_phrase != "":
                target_subtask_idx = get_target_subtask_id_v2(gt_subtasks, target_phrase, output_line, gt_actions)

                if target_subtask_idx != None:
                    if id_count >= len(grounding_result):
                        print(f"Warning: The number of grounding_result items does not match the number of output lines. id_count:{id_count}\t grounding_result:{grounding_result}")
                        break
                    pred_action_to_subtask_idx.append(target_subtask_idx)
                    pred_result = get_mask_iou_and_bbox_iou(grounding_result[id_count][0], gt_subtasks[target_subtask_idx]['target_label_id'], gt_subtasks[target_subtask_idx]['target_label_query_id'], mask)

                    pred_result["text"] = output_line
                    pred_results_info.append(pred_result)
                elif target_subtask_idx == None:
                    pass

            id_count += len(ref_count)
        else:
            pass

    assert len(pred_action_to_subtask_idx) == len(pred_results_info)
    return pred_action_to_subtask_idx, pred_results_info


def get_subtask_complete_info(gt_subtasks, subtask_index_seq_of_pred_actions, pred_results_idx):
    assert len(subtask_index_seq_of_pred_actions) == len(pred_results_idx)
    failure_reason_dict ={
        'no matched action':0,
        'failed to locate object':0,
        'violate subtask property':0,
    }
    num_subtask = len(gt_subtasks)
    conf_matrix = np.zeros((2, 3))
    subtask_complete_list = [False for i in range(len(gt_subtasks))]
    for i, subtask in enumerate(gt_subtasks):
        if i in subtask_index_seq_of_pred_actions:
            action_indexes = [index for index, value in enumerate(subtask_index_seq_of_pred_actions) if value == i]
            if subtask['type'] == "A":
                if len(action_indexes) == 1:
                    conf_matrix[0,0] += 1
                    if pred_results_idx[action_indexes[0]] == True:
                        subtask_complete_list[i] = True
                    else:
                        failure_reason_dict["failed to locate object"] += 1
                else:
                    failure_reason_dict["violate subtask property"] += 1
                    if len(action_indexes) == 2:
                        conf_matrix[0,1] += 1
                    else:
                        conf_matrix[0,2] += 1
                
            elif subtask["type"] == "B":
                if len(action_indexes) == 1:
                    failure_reason_dict["violate subtask property"] += 1
                    conf_matrix[1,0] += 1
                    if pred_results_idx[action_indexes[0]] == True:
                        subtask_complete_list[i] = True
                    else:
                        failure_reason_dict["failed to locate object"] += 1
                if len(action_indexes) == 2:
                    conf_matrix[1,1] += 1
                    left_index = action_indexes[0]
                    right_index = action_indexes[1]
                    B_time = subtask["time"]
                    A_time_sum = 0
                    for j in range(left_index+1,right_index):
                        A_time_sum += gt_subtasks[subtask_index_seq_of_pred_actions[j]]["time"]
                    if A_time_sum < B_time:
                        if pred_results_idx[left_index] == True and pred_results_idx[right_index] == True:
                            subtask_complete_list[i] = True
                        else:
                            failure_reason_dict["failed to locate object"] += 1
                    else:
                        failure_reason_dict["violate subtask property"] += 1
                else:
                    failure_reason_dict["violate subtask property"] += 1
                    conf_matrix[1,2] += 1

        else:
            failure_reason_dict["no matched action"] += 1
            if subtask['type'] == "A":
                conf_matrix[0,2] += 1
            elif subtask["type"] == "B":
                conf_matrix[1,2] += 1

    print(f"subtask_complete_list:{subtask_complete_list}")
    print(f"confusion matrix:\n{conf_matrix}")
    return subtask_complete_list, conf_matrix, failure_reason_dict
