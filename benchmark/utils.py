import os
import json
import re
import numpy as np
from models.metrics.taskscheduling_evaluator import get_target_subtask_id_v2, get_subtask_complete_info

import matplotlib.pyplot as plt
import seaborn as sns


def compute_max_time(subtasks):
    return sum(subtask["time"] for subtask in subtasks)

def max_sum_under_limit(nums, N):
    dp = [0] * (N) 
    choice = [-1] * (N) 

    for idx, num in enumerate(nums):
        for j in range(N - 1, num - 1, -1): 
            if dp[j - num] + num > dp[j]:  
                dp[j] = dp[j - num] + num
                choice[j] = idx 

    max_sum = 0
    for i in range(N - 1, -1, -1):  
        if dp[i] > max_sum:
            max_sum = dp[i]
            break  

    selected_indices = []
    j = max_sum
    while j > 0 and choice[j] != -1:
        selected_indices.append(choice[j])
        j -= nums[choice[j]] 

    selected_indices.reverse()  

    return selected_indices, max_sum

def minimize_time(subtasks):
    num_a_subtasks = 0
    num_subtasks = 0

    a_index_list = []
    a_time_list = []
    
    for i,subtask in enumerate(subtasks):
        num_subtasks += 1
        if subtask["type"] == "B":
            b_time = subtask["time"]
            b_index = i
        elif subtask["type"] == "A":
            num_a_subtasks += 1
            a_index_list.append(i)
            a_time_list.append(subtask["time"])
    if num_a_subtasks == num_subtasks:
        return a_index_list, sum(a_time_list)
    else:
        selected_indices, max_sum = max_sum_under_limit(a_time_list, b_time)
        all_indices = [i for i in range(len(a_time_list))]
        left_indices = list(set(all_indices) - set(selected_indices))
        if left_indices != []:
            additional_a = sum(a_time_list[i] for i in left_indices)
            b_time += additional_a
            [a_index_list[i] for i in selected_indices]
        if not left_indices:
            min_time_action_list = [b_index]+[a_index_list[i] for i in selected_indices]+[b_index]
        else:
            b_block = [b_index]+[a_index_list[i] for i in selected_indices]+[b_index]
            left_a_tasks = [a_index_list[i] for i in left_indices]
            insert_position = 0
            for i, task_idx in enumerate(left_a_tasks):
                if task_idx > b_index:
                    insert_position = i
                    break
            else:  
                insert_position = len(left_a_tasks)
            for i, item in enumerate(b_block):
                left_a_tasks.insert(insert_position + i, item)
            min_time_action_list = left_a_tasks
        return min_time_action_list, b_time



def compute_total_time(action_list, subtasks):
    max_time = compute_max_time(subtasks)
    record_task_id_set = set() 
    record_time = 0 
    in_b_time = 0
    b_start = False
    b_end = False
    count_b_time = False
    b_time = -1


    for act_task_id in action_list:
        
        if subtasks[act_task_id]["type"] == "A":
            if act_task_id not in record_task_id_set:
                record_task_id_set.add(act_task_id)
                if not b_start or b_end: 
                    record_time += subtasks[act_task_id]["time"]
                else: 
                    in_b_time += subtasks[act_task_id]["time"]
                    if in_b_time < b_time: 
                        pass
                    else: 
                        task_set = set(action_list)
                        max_time = sum(subtasks[i]["time"] for i in task_set)
                        return max_time 

            else: 
                pass

        elif subtasks[act_task_id]["type"] == "B":
            if not b_start:
                b_start = True
                have_b = True
                b_time = subtasks[act_task_id]["time"]
            elif b_start and not count_b_time: 
                b_end = True 
                record_time += subtasks[act_task_id]["time"] 
                count_b_time = True
    if b_start and not b_end:
        record_time += b_time
    return record_time


def calculate_efficiency_margin(gt_subtasks, subtask_index_seq_of_pred_actions):
    all_subtask_idx = set([i for i in range(len(gt_subtasks))])
    mentioned_subtask_idx = set(subtask_index_seq_of_pred_actions)
    left_subtask_idx = all_subtask_idx - mentioned_subtask_idx
    left_subtasks = [gt_subtasks[i] for i in list(left_subtask_idx)] 
    time_not_completed = sum(subtask["time"] for subtask in left_subtasks)

    complete_time_pred = compute_total_time(subtask_index_seq_of_pred_actions, gt_subtasks) 

    gt_max_time = compute_max_time(gt_subtasks)
    res, complete_time_opt = minimize_time(gt_subtasks)

    if gt_max_time == complete_time_opt: 
        return None
    efficiency_margin = complete_time_pred + time_not_completed - complete_time_opt
    if efficiency_margin < 0:
        exit()

    return efficiency_margin


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

import re
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def to_coco(kvs, keys):
    res = defaultdict(list)
    for k in keys:
        if k in kvs:
            caps = kvs[k]
            for c in caps:
                res[k].append({'caption': c})
        else:
            res[k].append({'caption': ''})
    return res


def print_dict(lan):
    for key in lan:
        print(f"{key}:      {lan[key]}")


def evaluate(ground_truths,prediction,verbose = True,iou = None):
    if iou is None:
        iou_25 = np.ones(len(ground_truths))
        iou_50 = np.ones(len(ground_truths))
    else:
        iou = np.array([iou[k] for k in ground_truths.keys()])
        iou_25 = iou>0.25
        print(f"iou 25: {np.sum(iou_25)/len(iou_25)}")
        iou_50 = iou>0.5
        print(f"iou 50: {np.sum(iou_50)/len(iou_50)}")
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]
    tokenizer = PTBTokenizer()
    ref_sent = ground_truths
    hypo_sent = prediction
    final_scores = {}
    ref_coco = tokenizer.tokenize(to_coco(ref_sent, ref_sent.keys()))
    hypo_coco = tokenizer.tokenize(to_coco(hypo_sent, ref_sent.keys()))
    for scorer, method in scorers:
        if verbose:
            print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(ref_coco, hypo_coco)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

def special_token_filter(lan,clean = True,truncation = True,max_length = 256):
    """
    Usage:
        clean the language, remove stop words and special tokens
    Args:
        lan: List[str], language to be cleaned
        clean: bool, if apply LEO clean strategy
        truncation: to avoid crash pycocoevalcap
    """
    replacements = {
    "ASSISTANT:": "",
    "ASSISTANT: ": "",
    "\n": "",
    "<s>": "",
    "</s>": "",
    "<unk>": "",
    "<p>": "",
    "</p>": "",
    "<GRU>": "",
    "<|endoftext|>": ""  
    }
    for old, new in replacements.items():
        lan = lan.replace(old, new)
    lan = lan.strip()
    lan = re.sub(r'\s{2,}', ' ', lan)
    if truncation:
        if len(lan)>max_length:
            lan = lan[:max_length]
    if clean:
        lan = clean_answer(lan)
    return lan



def ORS_language_evaluator(folder_path):
    taskscheduling_lan = {}
    taskscheduling_gt = {}

    idx_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for prediction_item in data['prediction']:
                    if 'taskscheduling' in prediction_item['type']:
                        output_language = prediction_item['output_language'].replace("</s>", "")
                        if "</think>" in output_language: 
                            output_language = output_language.split("</think>", 1)[1].strip()

                        taskscheduling_lan[idx_count]= [special_token_filter(output_language,clean=False)]
                        taskscheduling_gt[idx_count] = [special_token_filter(prediction_item["gt"],clean=False)]
                        idx_count+=1

    print(" ======================= Task Scheduling Evaluator ===================== ")
    print(f"Task Scheduling number: {len(taskscheduling_lan)}")
    if len(taskscheduling_gt):
        final_scores = evaluate(taskscheduling_gt,taskscheduling_lan,verbose=False)
        print_dict(final_scores)
    else:
        print(" can not find data to evaluate, skip task scheduling")






def visualize_distribution(iou_list, mask_list, time_list, save_name, manual_save_path):
    if not os.path.exists(manual_save_path):
        os.makedirs(manual_save_path)
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    sns.histplot(iou_list, bins=20, kde=True, ax=axs[0])
    axs[0].set_title('bbox IoU Distribution')
    sns.histplot(mask_list, bins=20, kde=True, ax=axs[1])
    axs[1].set_title('Mask Distribution')
    sns.histplot(time_list, bins=20, kde=True, ax=axs[2])
    axs[2].set_title('Efficiency margin Distribution')
    plt.savefig(f"{manual_save_path}/{save_name}.png")
    print(f"saved to {manual_save_path}/{save_name}.png")
    plt.close() 


def ORS_efficiency_and_grounding_evaluator(folder_path):
    count = 0
    abs_count = 0
    valid_count = 0

    throw_count = 0
    sum_subtask_compelte_rate_50 = 0
    sum_subtask_compelte_rate_25 = 0
    sum_grounding_25 = 0
    sum_grounding_50 = 0
    grounding_count = 0
    max_count = 10000 
    sum_efficiency_margin = 0
    sum_subtask_grounding_iou = 0
    sum_worest_efficiency_margin = 0

    sum_conf_matrix = np.zeros((2, 3))

    p_tag_pattern = r"<p>(.*?)</p>"
    ref_tag_pattern = r"<GRU>"


    error_on_task_num = {"sum_subtask_compelte_rate_25":{    "4":0,
                                                            "5":0,
                                                            "6":0,
                                                            "7":0,},
                        "subtask_count_25":{    "4":0,
                                                            "5":0,
                                                            "6":0,
                                                            "7":0,},                                    
                                                            
                        "sum_subtask_compelte_rate_50":{    "4":0,
                                                            "5":0,  
                                                            "6":0, 
                                                            "7":0,},
                        "subtask_count_50":{    "4":0,
                                                            "5":0,
                                                            "6":0,
                                                            "7":0,},  
                        
    }


    total_failure_reason_dict ={
        'no matched action':0,
        'failed to locate object':0,
        'violate subtask property':0,
    }

    total_failure_reason_dict_v2 ={
        'no matched action':0,
        'failed to locate object':0,
        'violate subtask property':0,
    }

    iou_list = []
    mask_list = []
    time_list = []

    match_1_box_ap25_all_counter = 0
    match_1_box_ap25_true_counter = 0
    match_2_query_match_all_counter = 0
    match_2_query_match_true_counter = 0
    sum_gt_grounding_25 = 0
    sum_gt_match2 = 0
    sum_gt_match2_top10 = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
                for prediction_item in data['prediction']:
                    if 'taskscheduling' in prediction_item['type']:
                        gt_subtasks = prediction_item['subtasks']
                        abs_count +=1 
                        completed_subtask = 0
                        pred_results_idx = []

                        actions_to_subtask_idx_list = prediction_item['tasksheduling_result']["actions_to_subtask_idx_list"]
                        pred_results_idx_list = prediction_item['tasksheduling_result']["pred_results_idx_list"]
                        input_language = prediction_item['input_language']

                        output_language = prediction_item['output_language'].replace("</s>", "")
                        gt_subtasks = prediction_item['subtasks']

                        if "<think>" and "</think>" in output_language:
                            think_content = output_language.split("</think>", 1)[0].strip()
                            output_language = output_language.split("</think>", 1)[1].strip()
                        else:
                            think_content = None

                        gt = prediction_item['gt']
                        grounding_result = prediction_item['grounding_result'] 
                        query_ids_answer = prediction_item['gt_inst_ids']['query_ids_answer'] 
                        
                        gt_actions = prediction_item['actions']
                        GT_actions_to_subtask_idx_list = [item["subtask_index"] for item in gt_actions]

                        for idx, action in enumerate(gt_actions):
                            action_target_id = action['target_id']
                            for subtask in gt_subtasks:
                                if subtask['target_id'] == action_target_id:
                                    subtask['target_query_id'] = query_ids_answer[idx][0]  
                                    break  

                        output_lines = output_language.split("\n")
                        pred_actions_to_subtask_idx_list = [item+1 for item in actions_to_subtask_idx_list]

                
                        if grounding_result == None:
                            for subtask in gt_subtasks:
                                if subtask['type'] == "B":
                                    gt_max_time = compute_max_time(gt_subtasks)
                                    res, complete_time_opt = minimize_time(gt_subtasks)
                                    if gt_max_time > complete_time_opt:
                                        efficiency_margin = gt_max_time - complete_time_opt
                                        sum_efficiency_margin += efficiency_margin
                                        valid_count +=1 
                                        time_list.append(efficiency_margin)
                                        sum_worest_efficiency_margin += gt_max_time - complete_time_opt 
                            continue 
                        

                        id_count = 0
                        for i in range(len(output_lines)): 
                            output_line = output_lines[i]
                            if "<GRU>" in output_line: 
                                target_phrases = re.findall(p_tag_pattern, output_line)
                                target_phrase = " ".join(target_phrases) if target_phrases else ""
                                target_phrase = re.sub(ref_tag_pattern, '', target_phrase).strip()
                                if target_phrase != "":
                                    target_subtask_idx = get_target_subtask_id_v2(gt_subtasks, target_phrase, output_line, gt_actions) 
                                    
                                    
                                    if target_subtask_idx != None:
                                        if gt_subtasks[target_subtask_idx]["label"] not in output_line:
                                            total_failure_reason_dict['no matched action'] += 1
                                        if id_count > len(pred_results_idx_list)-1:
                                            continue
                                        id_count += 1
                                

                        bbox_iou_greater_than_0_5 = [item["bbox_iou"] > 0.5 for item in pred_results_idx_list]
                        bbox_iou_greater_than_0_25 = [item["bbox_iou"] > 0.25 for item in pred_results_idx_list]


                        efficiency_margin = calculate_efficiency_margin(gt_subtasks, actions_to_subtask_idx_list) 
                        if efficiency_margin != None:
                            sum_efficiency_margin += efficiency_margin
                            res, complete_time_opt = minimize_time(gt_subtasks)
                            max_operation_time = compute_max_time(gt_subtasks)
                            if efficiency_margin > 0:
                                total_failure_reason_dict['violate subtask property'] += 1
                            sum_worest_efficiency_margin += max_operation_time - complete_time_opt 
                            valid_count +=1
                            time_list.append(efficiency_margin)


                        subtask_complete_list, conf_matrix, failure_reason_dict= get_subtask_complete_info(gt_subtasks, actions_to_subtask_idx_list, bbox_iou_greater_than_0_25)
                        subtask_compelte_rate_25 = subtask_complete_list.count(True)/len(subtask_complete_list)
                        sum_subtask_compelte_rate_25 += subtask_compelte_rate_25
                        iou_list.append(subtask_compelte_rate_25)

                        error_on_task_num["sum_subtask_compelte_rate_25"][str(len(gt_subtasks))] += subtask_compelte_rate_25
                        error_on_task_num["subtask_count_25"][str(len(gt_subtasks))] += 1
                        total_failure_reason_dict['no matched action'] += failure_reason_dict['no matched action']
                        total_failure_reason_dict['failed to locate object'] += failure_reason_dict['failed to locate object']
                        total_failure_reason_dict['violate subtask property'] += failure_reason_dict['violate subtask property']

                        sum_grounding_25 += bbox_iou_greater_than_0_25.count(True)/len(bbox_iou_greater_than_0_25) if len(bbox_iou_greater_than_0_25) != 0 else 0
                        sum_grounding_50 += bbox_iou_greater_than_0_5.count(True)/len(bbox_iou_greater_than_0_5) if len(bbox_iou_greater_than_0_5) != 0 else 0
                        
                        grounding_count += len(bbox_iou_greater_than_0_25)
                        subtask_complete_list, conf_matrix, failure_reason_dict = get_subtask_complete_info(gt_subtasks, actions_to_subtask_idx_list, bbox_iou_greater_than_0_5)
                        subtask_compelte_rate_50 = subtask_complete_list.count(True)/len(subtask_complete_list)
                        sum_subtask_compelte_rate_50 += subtask_compelte_rate_50
                        error_on_task_num["sum_subtask_compelte_rate_50"][str(len(gt_subtasks))] += subtask_compelte_rate_50
                        error_on_task_num["subtask_count_50"][str(len(gt_subtasks))] += 1

                        sum_conf_matrix+=conf_matrix
                        mask_iou_values = [entry['mask_iou'] for entry in pred_results_idx_list]
                        average_mask_iou = sum(mask_iou_values) / len(pred_results_idx_list) if len(bbox_iou_greater_than_0_25) != 0 else 0 

                        mask_list.append(average_mask_iou)

                        sum_subtask_grounding_iou += average_mask_iou
                        count += 1
    mean_worest_efficiency_margin = sum_worest_efficiency_margin/valid_count
    mean_efficiency_margin = sum_efficiency_margin/valid_count


    print(f"Grounding bbox AP@0.25:{sum_grounding_25/abs_count}")
    print(f"Time Efficiency:{(mean_worest_efficiency_margin-mean_efficiency_margin)/mean_worest_efficiency_margin}")
    print(f"Grounding mask IoU:{sum_subtask_grounding_iou/abs_count}")
