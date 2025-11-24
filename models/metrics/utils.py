import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict, Counter
import copy
from datasets.utils import read_axis_align_matrix

from models.misc import get_batch_aabb_pair_ious

def get_mask_iou(gt_masks,pred_masks):
    inter = (gt_masks & pred_masks).sum()
    outer = (gt_masks | pred_masks).sum()
    iou = inter / (outer + 1e-8)
    return iou


def eval_seg_model(bid,
                   config,
                   extra_lang,
                   full_res_coords,
                   raw_masks,
                   raw_heatmap,
                   target_full_res,
                   pred_lang_logits,
                   training,
                   ):
    assert bid == 0
    lang_phrase = extra_lang.raw_phrases[bid]
    flatten_lang_token_inst_id_pair = extra_lang.flatten_lang_token_inst_id_pairs[bid]
    raw_lang_type = extra_lang.raw_lang_types[bid]
    lang_texts = ''.join(extra_lang.batch_concat_texts[0:extra_lang.batch_num_concat_texts[bid]]).split('. ')[:-1] # remove last

    assert len(lang_texts) == len(flatten_lang_token_inst_id_pair) == len(raw_lang_type), 'The separated text num does not match text token'

    gt_extra_masks = []
    gt_extra_query_texts = []
    pred_extra_masks = []
    pred_extra_masks_instance_coordscore = []
    each_ious = []
    eval_types = []

    # get axis aligned coordinates
    if not getattr(config.model, 'axis_align_coord', False): # not perform axis align in dataloader
        raise NotImplementedError
        axis_align_matrix = read_axis_align_matrix(f"./data/rawscannet/scans/{scene_id}/{scene_id}.txt")
        aligned_scene_coordinates = full_res_coords[bid] @ axis_align_matrix[:3, :3].T + axis_align_matrix[:3, 3:4].T # same to mesh.transform
    else:
        aligned_scene_coordinates = full_res_coords[bid]
    
    pred_lang_logits = pred_lang_logits
    start_token = (config.model.num_classes - 1) if not config.data.sample_class_labels or not training else 0
    each_gt_bbox_list = []
    each_pred_bbox_list = []
    each_pred_bbox_is_none = []
    each_gt_bbox_ious = []

    each_iou_25_f1_score = []
    each_iou_50_f1_score = []

    for i in range(start_token, len(lang_texts)):
        eval_type = raw_lang_type[i]
        all_unique_token_ids = np.unique([token_bid for token_bid, gt_inst_id in flatten_lang_token_inst_id_pair[i]])
        if len(all_unique_token_ids) == 0: # TODO(need to ensure all texts should has at least one nouns!!!)
            continue
        each_ious.append([])
        each_gt_bbox_list.append([])
        each_pred_bbox_list.append([])
        each_pred_bbox_is_none.append([])

        each_iou_25_f1_score.append([])
        each_iou_50_f1_score.append([])
        
        for now_token_id in all_unique_token_ids:
            inst_ids = np.asarray([gt_inst_id for token_bid, gt_inst_id in flatten_lang_token_inst_id_pair[i] if token_bid == now_token_id], dtype=int)

            # GT phrases box
            gt_extra_masks.append( target_full_res[bid]['masks'].numpy()[ inst_ids ].sum(axis=0)>0 ) # concate all gt insts
            gt_extra_query_texts.append((lang_phrase[now_token_id], lang_texts[i]))

            # Pred phrases
            pos_token_ids = now_token_id-start_token
            pred_lang_logits_i = pred_lang_logits[:, pos_token_ids]
    
            # eval for different data type
            # ============== only single object ==================
            if 'scanrefer' in eval_type or 'taskrefer' in eval_type:
                top1_mask = torch.zeros((pred_lang_logits.shape[0],), dtype=bool, device='cuda')
                top1_mask[pred_lang_logits_i.argmax()] = 1
                pred_lang_logits_i = pred_lang_logits_i * top1_mask
                pred_query_ids = (pred_lang_logits_i > 1e-3).cpu().numpy()
            # ============== multi objects ==================
            else: # m3dref; groundedscenecaption
                pred_query_ids = (pred_lang_logits_i > 0.3).cpu().numpy() # [150, ] bool
                
            # ============== m3d bbox iou ===================
            # for every pred mask
            if pred_query_ids.sum()==0: # no prediction
                if inst_ids.shape[0]: # have gt
                    each_iou_25_f1_score[-1].append(0.0)
                    each_iou_50_f1_score[-1].append(0.0)
                else:               # no gt
                    each_iou_25_f1_score[-1].append(1.0)
                    each_iou_50_f1_score[-1].append(1.0)
            else: # have prediction
                if not inst_ids.shape[0]: # no gt
                    each_iou_25_f1_score[-1].append(0.0)
                    each_iou_50_f1_score[-1].append(0.0)
                else:
                    try:
                        m3d_all_pred_bbox = []
                        for qi in np.where(pred_query_ids)[0]:
                            m3d_pred_mask =(raw_masks[:, qi]>0)
                            m3d_pred_points = aligned_scene_coordinates[m3d_pred_mask.numpy().astype(bool)]
                            if m3d_pred_points.size:
                                m3d_min_vals,m3d_max_vals  = m3d_pred_points.min(axis=0),m3d_pred_points.max(axis=0)
                                m3d_pred_bbox = np.vstack((m3d_min_vals, m3d_max_vals))
                                m3d_all_pred_bbox.append(m3d_pred_bbox)
                        m3d_all_gt_bbox = []
                        # for every gt mask
                        for gt_mask_id in inst_ids:
                            m3d_gt_mask = target_full_res[bid]['masks'].numpy()[ gt_mask_id ]
                            m3d_gt_points = aligned_scene_coordinates[m3d_gt_mask.astype(bool)].squeeze()
                            if m3d_gt_points.size:
                                m3d_min_vals,m3d_max_vals  = m3d_gt_points.min(axis=0),m3d_gt_points.max(axis=0)
                                m3d_gt_bbox = np.vstack((m3d_min_vals, m3d_max_vals))
                                m3d_all_gt_bbox.append(m3d_gt_bbox)
                        if m3d_all_gt_bbox and m3d_all_pred_bbox:
                            m3d_all_pred_bbox = torch.tensor(np.stack(m3d_all_pred_bbox,axis=0))
                            m3d_all_gt_bbox = torch.tensor(np.stack(m3d_all_gt_bbox,axis=0))
                            M, N = m3d_all_pred_bbox.size(0), m3d_all_gt_bbox.size(0)
                            m3d_all_pred_bbox_expanded = m3d_all_pred_bbox.unsqueeze(1).expand(M, N, 2, 3)
                            m3d_all_pred_bbox_repeated = m3d_all_pred_bbox_expanded.reshape(M*N, 2, 3)
                            m3d_all_gt_bbox_expanded = m3d_all_gt_bbox.unsqueeze(0).expand(M, N, 2, 3)
                            m3d_all_gt_bbox_repeated = m3d_all_gt_bbox_expanded.reshape(M*N, 2, 3)
                            m3d_bbox_iou = np.array(get_batch_aabb_pair_ious(m3d_all_pred_bbox_repeated, m3d_all_gt_bbox_repeated).view(M, N))
                            m3d_max_dim = max(M, N)
                            m3d_padded_array = np.zeros((m3d_max_dim, m3d_max_dim))
                            m3d_padded_array[:M, :N] = m3d_bbox_iou
                            m3d_bbox_iou = m3d_padded_array
                            m3d_row_idx, m3d_col_idx = linear_sum_assignment(m3d_bbox_iou * -1)
                            iou_25_tp = 0
                            iou_50_tp = 0
                            # iterate matched pairs, check ious
                            for ii in range(M):
                                m3d_iou = m3d_bbox_iou[m3d_row_idx[ii], m3d_col_idx[ii]]
                                # calculate true positives
                                if m3d_iou >= 0.25:
                                    iou_25_tp += 1
                                if m3d_iou >= 0.5:
                                    iou_50_tp += 1
                            # calculate precision, recall and f1-score for the current scene
                            each_iou_25_f1_score[-1].append(2 * iou_25_tp / (N+M))
                            each_iou_50_f1_score[-1].append(2 * iou_50_tp / (N+M))
                        elif m3d_all_gt_bbox and not m3d_all_pred_bbox:
                            each_iou_25_f1_score[-1].append(0.0)
                            each_iou_50_f1_score[-1].append(0.0)
                        elif not m3d_all_gt_bbox and m3d_all_pred_bbox:
                            each_iou_25_f1_score[-1].append(0.0)
                            each_iou_50_f1_score[-1].append(0.0)
                        elif not m3d_all_gt_bbox and not m3d_all_pred_bbox:
                            each_iou_25_f1_score[-1].append(1.0)
                            each_iou_50_f1_score[-1].append(1.0)
                    except Exception as e:
                        print(e)
                        from IPython import embed;embed()

            # =============== compute masks for visualization =============
            instance_center_coords = []
            for qi in np.where(pred_query_ids)[0]:
                pred_mask = (raw_masks[:, qi]>0).to(float)
                coord = full_res_coords[bid][pred_mask.numpy().astype(bool)].mean(axis=0)
                mask_score = (raw_heatmap[:, qi].sigmoid() * pred_mask).sum(0) / (pred_mask.sum(0) + 1e-6)
                score = (pred_lang_logits_i[qi] * mask_score).cpu().numpy()
                instance_center_coords.append( np.append(coord, score) )
            if len(instance_center_coords) > 0:
                instance_center_coords = np.stack(instance_center_coords, axis=0)
            else:
                instance_center_coords = np.zeros((0, 4), dtype=np.float32)

            # ================== compute box =====================
            pred_extra_masks.append((raw_masks[:, pred_query_ids].cpu().numpy().sum(axis=1) > 0))
            pred_extra_masks_instance_coordscore.append( instance_center_coords )

            # ================== compute box iou =====================
            gt_points = aligned_scene_coordinates[gt_extra_masks[-1].astype(bool)]
            pred_points = aligned_scene_coordinates[pred_extra_masks[-1].astype(bool)]

            min_vals, max_vals = gt_points.min(axis=0), gt_points.max(axis=0)
            gt_bbox = np.vstack((min_vals, max_vals))
            each_gt_bbox_list[-1].append(gt_bbox)
            
            pred_bbox = None
            if pred_bbox is None:
                # use the mask to obtain box
                if pred_points.size > 0:
                    min_vals,max_vals = pred_points.min(axis=0),pred_points.max(axis=0)
                    pred_bbox = np.vstack((min_vals, max_vals))
                    each_pred_bbox_list[-1].append(pred_bbox)
                    each_pred_bbox_is_none[-1].append(False)
                else:
                    each_pred_bbox_list[-1].append(gt_bbox)
                    each_pred_bbox_is_none[-1].append(True)
            else:
                # if there is a reg branch
                each_pred_bbox_list[-1].append(
                    np.vstack((pred_bbox[:3], pred_bbox[3:]))
                )
                each_pred_bbox_is_none[-1].append(False)

            inter = (gt_extra_masks[-1] & pred_extra_masks[-1]).sum()
            outer = (gt_extra_masks[-1] | pred_extra_masks[-1]).sum()
            iou = inter / (outer + 1e-8)
            each_ious[-1].append(iou)
        
        each_iou_25_f1_score[-1] = np.mean(each_iou_25_f1_score[-1])
        each_iou_50_f1_score[-1] = np.mean(each_iou_50_f1_score[-1])

        assert len(each_gt_bbox_list[-1]) == len(each_pred_bbox_list[-1])and len(each_pred_bbox_list[-1]) == len(each_ious[-1])
        if len(each_gt_bbox_list[-1]) > 0:
            gt_bboxes,pred_bboxes = torch.tensor(np.stack(each_gt_bbox_list[-1],axis=0)),torch.tensor(np.stack(each_pred_bbox_list[-1],axis=0))
            bbox_ious_with_none = get_batch_aabb_pair_ious(gt_bboxes,pred_bboxes).tolist()
            each_bbox_ious = []
            for j, (bbox_iou, is_none) in enumerate(zip(bbox_ious_with_none, each_pred_bbox_is_none[-1])):
                if is_none:
                    each_bbox_ious.append(0.0)
                    each_pred_bbox_list[-1][j] = None
                else:
                    each_bbox_ious.append(bbox_iou)
            each_gt_bbox_ious.append(float(np.mean(np.asarray(each_bbox_ious))))
            each_ious[-1] = np.mean(each_ious[-1])
        else:
            each_gt_bbox_ious.append(0.0)
            each_ious[-1] = 0.0

        eval_types.append(eval_type)

    return  (np.asarray(each_ious), np.asarray(eval_types), np.asarray(each_gt_bbox_ious), np.asarray(each_iou_25_f1_score), np.asarray(each_iou_50_f1_score)),\
        gt_extra_masks,gt_extra_query_texts,pred_extra_masks,pred_extra_masks_instance_coordscore

def merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        if dict1[key] is not None and dict2[key] is not None:
            try:
                merged_dict[key] = dict1[key] + dict2[key]
            except:
                merged_dict[key] = dict2[key]
        elif dict1[key] is not None:
            merged_dict[key] = dict1[key] 
        else:
            merged_dict[key] = dict2[key]
    return merged_dict
    
def calculate_AP_score(data, thresholds = [0.25, 0.5]):
    for key in data:
        if len(data[key]):
            if not 'm3dref' in key:
                ious = np.array([x for x,y in data[key]])
                bbox_ious = np.array([y for x,y in data[key]])
                averages = np.array([[(ious > threshold).sum() / ious.shape[0] for threshold in thresholds],
                            [(bbox_ious > threshold).sum() / bbox_ious.shape[0] for threshold in thresholds]])
                data[key] = averages
            else:
                iou_25_f1_score = np.array([x for x,y in data[key]])
                iou_50_f1_score = np.array([y for x,y in data[key]])
                averages = np.array([ iou_25_f1_score.mean(), iou_50_f1_score.mean() ])
                data[key] = averages
        else:
            if not 'm3dref' in key:
                data[key] = np.zeros((2,len(thresholds))) - 1
            else:
                data[key] = np.zeros((2,)) - 1
    if 'donteval' in data:
        data['donteval'] = None
    return data

def collect_grounding_score(all_preds,ap_results,log_prefix):
    all_gt_ious = []
    all_gt_bbox_ious = []
    all_eval_types = []
    all_iou_25_f1_score = []
    all_iou_50_f1_score = []
    
    template = OrderedDict([
        ("scanrefer:unique", []), 
        ("scanrefer:multiple", []), 
        ("scanrefer:overall", []), 
        ("m3dref:overall", []), 
        ("donteval", []), 
        ("taskrefer:ScanNet", []), 
        ("taskrefer:3RScan", []), 
        ("taskrefer:MultiScan", []), 
        ("taskrefer:ARKitScenes", []), 
        ("taskrefer:HM3D", []), 
        ("taskrefer:overall", []), 
    ])

    valid_datasets = ['ScanNet', '3RScan', 'MultiScan', 'ARKitScenes', 'HM3D']

    all_everry_type_average_by_scene = copy.deepcopy(template)
    for key in all_everry_type_average_by_scene:
        all_everry_type_average_by_scene[key]=np.zeros((2,3))
    all_every_type_score = copy.deepcopy(template)
    count_score_all_scene = np.zeros(len(template))
    for key in all_preds.keys():
        dataset_from_key = key.split('_')[0] if '_' in key else None
        if dataset_from_key not in valid_datasets:
            dataset_from_key = 'ScanNet'  
            
        count_score_pre_scene = np.zeros(len(template))
        every_type_score = copy.deepcopy(template)
        all_gt_ious.append(all_preds[key]['gt_ious'][0].astype(float))
        all_iou_25_f1_score.append(all_preds[key]['gt_ious'][3].astype(float))
        all_iou_50_f1_score.append(all_preds[key]['gt_ious'][4].astype(float))
        all_eval_types.append(all_preds[key]['gt_ious'][1])
        all_gt_bbox_ious.append(all_preds[key]['gt_ious'][2].astype(float))
        for i, (gt_iou,gt_bbox_iou,eval_type) in enumerate(zip(all_gt_ious[-1],all_gt_bbox_ious[-1],all_eval_types[-1])):
            if gt_iou == np.nan:
                gt_iou = 0
            dataset_name = eval_type.split(":")[0]
            if dataset_name == "scanrefer":
                if "multiple" in eval_type:
                    every_type_score["scanrefer:multiple"].append([gt_iou,gt_bbox_iou])
                    count_score_pre_scene[1]+=1
                else:
                    every_type_score["scanrefer:unique"].append([gt_iou,gt_bbox_iou])
                    count_score_pre_scene[0]+=1
                every_type_score["scanrefer:overall"].append([gt_iou,gt_bbox_iou])
                count_score_pre_scene[2]+=1
            elif dataset_name == 'm3dref':
                every_type_score["m3dref:overall"].append([all_iou_25_f1_score[-1][i], all_iou_50_f1_score[-1][i]])
                count_score_pre_scene[3]+=1
            elif dataset_name == "taskrefer":

                taskrefer_key = f"taskrefer:{dataset_from_key}"
                every_type_score[taskrefer_key].append([gt_iou, gt_bbox_iou])
                

                dataset_index = list(template.keys()).index(taskrefer_key)
                count_score_pre_scene[dataset_index] += 1
                

                every_type_score["taskrefer:overall"].append([gt_iou, gt_bbox_iou])
                count_score_pre_scene[10] += 1
            else:
                every_type_score['donteval'].append([gt_iou,gt_bbox_iou])
                count_score_pre_scene[4]+=1
        count_score_all_scene += count_score_pre_scene>0
        all_every_type_score = merge_dicts(all_every_type_score,every_type_score)
        every_type_score = calculate_AP_score(every_type_score)
        
    assert len(all_gt_ious[0])==len(all_gt_bbox_ious[0])

    all_every_type_score = calculate_AP_score(all_every_type_score)

    all_gt_ious = np.concatenate(all_gt_ious)
    if len(all_gt_ious) > 0:
        ap_results[f'{log_prefix}_mean_grounding_iou_25'] = (all_gt_ious > 0.25).sum() / all_gt_ious.shape[0]
        ap_results[f'{log_prefix}_mean_grounding_iou_50'] = (all_gt_ious > 0.5).sum() / all_gt_ious.shape[0]

        for key in all_every_type_score.keys():
            if key == "donteval":
                continue
            if not 'm3dref' in key:
                if all_every_type_score[key][0,0] < -0.1:
                    continue
                ap_results[f'{log_prefix}_mean_{key}_grounding_iou_25'] = all_every_type_score[key][0,0]
                ap_results[f'{log_prefix}_mean_{key}_grounding_iou_50'] = all_every_type_score[key][0,1]
                ap_results[f'{log_prefix}_mean_{key}_grounding_bbox_iou_25'] = all_every_type_score[key][1,0]
                ap_results[f'{log_prefix}_mean_{key}_grounding_bbox_iou_50'] = all_every_type_score[key][1,1]
            else:
                ap_results[f'{log_prefix}_mean_{key}_grounding_bbox_iou_25_f1_score'] = all_every_type_score[key][0]
                ap_results[f'{log_prefix}_mean_{key}_grounding_bbox_iou_50_f1_score'] = all_every_type_score[key][1]
    return ap_results
