import numpy as np
import json
from pathlib import Path
from collections import OrderedDict
import os.path as osp

DEBUG = False

lang_root_path = Path('./data/langdata/')
lang_reformat_path = Path('./data/langdata/reformat/')
lang_reformat_path.mkdir(parents=True, exist_ok=True)

with open('./data/scannet_inst_ids.json') as f:
    scene_filter_inst_ids = json.load(f)

# As the grounding text often contains multiple `.`, we replace all intermediate end punctuation with ";" and only keep the last end punctuation.
def replace_punctuation(s):
    import re
    # Replace all end punctuation with ";"
    s = re.sub(r'([.!?])', r';', s)
    # Replace the last ";" with "."
    s = re.sub(r';(\s*)$', r'.', s)

    if not re.search(r'\.$', s): # if not end with ".", add "." # must ensure the last char is .
        s = re.sub(r'(\S)(\s*)$', r'\1.', s)
    return s

# ----------------------- ScanRefer ----------------------------
if not osp.exists(lang_reformat_path / 'scanrefer_format.json'):
    scanrefer_path = lang_root_path / 'scanrefer/ScanRefer_filtered_full_withroot_addeval.json'
    with open(scanrefer_path) as f:
        scanrefer_source = json.load(f)
    filter_scanrefer_source = []
    for i, lang in enumerate(scanrefer_source):
        if not lang['all_phrases_positions']:
            continue
        lang['all_phrases_positions'] = [k for k in lang['all_phrases_positions'] if k]
        if not lang['all_phrases_positions']:
            if DEBUG:
                print(f'skip scanrefer {lang["scene_id"]}:{i}')
            continue
        lang['lang_type'] = 'scanrefer:' + lang['eval_type']
        lang['object_ids'] = [[int(lang['object_id'])] for i in lang['all_phrases_positions']] # repeat for each noun phrases
        lang.pop('object_id')
        lang.pop('token') 
        lang['description'] = replace_punctuation(lang['description'])
        filter_scanrefer_source.append(lang)
    with open(lang_reformat_path / 'scanrefer_format.json', 'w') as f:
        json.dump(filter_scanrefer_source, f)
    
    print(f'ScanRefer {len(filter_scanrefer_source)} data items Processed.')

# ----------------------- Multi3DRef ---------------------------
if not osp.exists(lang_reformat_path / 'm3dref_format.json'):
    m3dref_path = './data/langdata/multi3drefer/m3drefer_train+val_add_positives_complete.json'
    with open(m3dref_path) as f:
        m3dref_source = json.load(f)
    for lang in m3dref_source:
        if not lang['all_phrases_positions']:
            continue
        lang['all_phrases_positions'] = [k for k in lang['all_phrases_positions'] if k]
        if not lang['all_phrases_positions']:
            if DEBUG:
                print(f'skip m3dref {lang["scene_id"]}:{i}')
            continue
        lang['lang_type'] = 'm3dref:' + lang['eval_type'] + ':' + str(lang['ann_id']) # attach the ann_id at last
        lang['object_ids'] = [lang['object_ids'] for i in lang['all_phrases_positions']] # repeat for each noun phrases
        if 'token' in lang: lang.pop('token') 
        if 'tokens' in lang: lang.pop('tokens') 
        lang['description'] = replace_punctuation(lang['description'])
    with open(lang_reformat_path / 'm3dref_format.json', 'w') as f:
        json.dump(m3dref_source, f)
    print(f'Multi3DRef: {len(m3dref_source)} data items Processed.')

# ----------------------- Grounded Scene Caption -----------------
if not osp.exists(lang_reformat_path / 'groundedscenecaption_format.json'):
    scene_lang_source = json.load(open('./data/langdata/GPTsummary/all_filtered_step2_captions_by_scene_v2_haifeng.json'))
    filtered_scene_lang_source = []

    for i, lang in enumerate(scene_lang_source):
        valid_flag = True
        lang['description'] = lang['sentence']; lang.pop('sentence', None)
        lang['all_phrases_positions'] = lang['positive']; lang.pop('positive', None)
        lang.pop('response_step_2')
        lang.pop('refined_caption')

        positive_chars = []
        for object_ids in lang['all_phrases_positions']:
            for p in lang['all_phrases_positions'][object_ids]:
                positive_chars.append((p[0], p[1], int(object_ids)))
        positive_chars = sorted(positive_chars)
        positive_dict = OrderedDict()
        for p0, p1, obj_id in positive_chars:
            if (p0, p1) not in positive_dict:
                positive_dict[(p0, p1)] = [obj_id]
            else:
                positive_dict[(p0, p1)].append(obj_id)
        
        lang['all_phrases_positions'] = []
        lang['object_ids'] = []
        for positives, inst_ids in positive_dict.items():
            if not set(inst_ids).issubset(set( scene_filter_inst_ids[lang['scene_id']])):
                valid_flag = False
                if DEBUG:
                    print(f'{i}: {lang["scene_id"]} uses instance_ids outside the annotation!')
            # Filtered by gt_inst_ids
            inst_ids = list(set(inst_ids) & set( scene_filter_inst_ids[lang['scene_id']]))
            if len(inst_ids) == 0: continue
            lang['all_phrases_positions'].append(positives)
            lang['object_ids'].append(inst_ids)

        # for p, oid in zip(lang['all_phrases_positions'], lang['object_ids']):
        #     print(lang['description'][p[0]:p[1]], oid)
        
        lang['object_name'] = 'groundedscenecaption'
        lang['lang_type'] = 'groundedscenecaption:'

        # remove the intermediate None type
        if not lang['all_phrases_positions']:
            continue
        all_phrases_positions = []
        for j, jpp in enumerate(lang['all_phrases_positions']): # remove intermediate None
            if jpp:
                all_phrases_positions.append(jpp)
        lang['all_phrases_positions'] = all_phrases_positions
        lang['description'] = replace_punctuation(lang['description'])
        
        if valid_flag:
            filtered_scene_lang_source.append(lang)

    with open(lang_reformat_path / 'groundedscenecaption_format.json', 'w') as f:
        json.dump(filtered_scene_lang_source, f)

    print(f'GroundedSceneCaption: {len(filtered_scene_lang_source)} data items Processed.')

# ----------------------- ScanQA ------------------------------
if not osp.exists(lang_reformat_path / 'scanqa_format.json'):
    scanqa_lang_source = json.load(open('./data/langdata/ScanQA_v1.0/scanQA_gpt_processv2.json'))
    scanqa_lang_source.extend(json.load(open('./data/langdata/ScanQA_v1.0/scanQA_val_gpt_processv2.json')))

    for lang in scanqa_lang_source:
        lang['answer'] = lang['raw_data']['answers'][lang['idx']]
        lang['question'] = lang['raw_data']['question']
        lang['raw_object_names'] = lang['raw_data']['object_names']
        lang['raw_object_ids'] = lang['raw_data']['object_ids']

        positive_chars = []
        for object_id in lang['positive']:
            for p in lang['positive'][object_id]:
                positive_chars.append((p[0], p[1], int(object_id)))
        positive_chars = sorted(positive_chars)
        positive_dict = OrderedDict()
        for p0, p1, obj_id in positive_chars:
            if (p0, p1) not in positive_dict:
                positive_dict[(p0, p1)] = [obj_id]
            else:
                positive_dict[(p0, p1)].append(obj_id)
        
        lang['all_phrases_positions'] = []
        lang['object_ids'] = []
        for positives, inst_ids in positive_dict.items():
            # Filtered by gt_inst_ids
            inst_ids = list(inst_ids)
            if len(inst_ids) == 0: continue
            lang['all_phrases_positions'].append(positives)
            lang['object_ids'].append(inst_ids)
        
        lang['lang_type'] = 'scanqa:' + lang['raw_data']['question_id'] + ':' + str(lang['idx']) + ':'
        lang.pop('raw_data')

    with open(lang_reformat_path / 'scanqa_format.json', 'w') as f:
        json.dump(scanqa_lang_source, f)

# ----------------------- Object description ------------------------------
if not osp.exists(lang_reformat_path / 'objectdescription_format.json'):
    with open('data/langdata/GPTsummary/all_objects_by_scene.json') as f:
        object_description = json.load(f)
    with open('data/langdata/GPTsummary/all_objects_by_scene_val.json') as f:
        object_description_val = json.load(f)
    for scene_id in object_description_val:
        for obj_desc in object_description_val[scene_id]:
            obj_desc['description'] = obj_desc['caption']
            obj_desc['scan_id'] = scene_id
            obj_desc.pop('caption')
        object_description_val[scene_id] = dict(object_list=object_description_val[scene_id])
    object_description.update(object_description_val)

    object_description_source = []
    for scene_id, scene_obj_descs in object_description.items():
        for obj_desc in scene_obj_descs['object_list']:
            assert obj_desc['scan_id'] == scene_id
            if obj_desc['object_id'] not in scene_filter_inst_ids[obj_desc['scan_id']]:
                # print(obj_desc['object_name'])
                continue
            if 'object_name' in obj_desc and ('wall' in obj_desc['object_name'].lower() or 'floor' in obj_desc['object_name'].lower() or 'ceiling' in obj_desc['object_name'].lower()):
                continue
            if 'description' in obj_desc and obj_desc['description'] is not None:
                qa_dict = dict(
                    scene_id=scene_id,
                    answer=obj_desc['description'],
                    object_ids=[[obj_desc['object_id']]],
                    # object_name=obj_desc['object_name'],
                    lang_type='objdesc:',
                )
                object_description_source.append(qa_dict)

    with open(lang_reformat_path / 'objectdescription_format.json', 'w') as f:
        json.dump(object_description_source, f)

# -------------------- 3D LLM data ---------------------------------
if not osp.exists(lang_reformat_path / '3dllm_format.json'):
    data_3dllm_source = []
    data_list_3dllm = [
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_dialogue_filtered_train.json',
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_dialogue_filtered_val.json',
                #  'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_planning_filtered_train.json',
                #  'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_planning_filtered_val.json',
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_question_answer_train.json',
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_embodied_question_answer_val.json',
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_scene_description_train.json',
                    'data/langdata/3D_LLM_processedLL3DA/3d_llm_scene_description_val.json',
                    ]
    for d in data_list_3dllm:
        data_3dllm_source.extend(json.load(open(d)))
    for d in data_3dllm_source:
        scene_id = d['scene_id']
        d['lang_type'] = '3dllm:'
        d['question'] = d['question'].replace('### human:', 'USER:').replace('### assistant:', 'ASSISTANT:')
        assert d['question'].startswith('USER: ') and d['question'].endswith('ASSISTANT:'), f"invliad question 3D-LLM: {d['question']}"
        d['question'] = d['question'][len('USER:'):-len('ASSISTANT:')]
        assert len(d['answers']) == 1, d
        d['answers'] = d['answers'][0]

    with open(lang_reformat_path / '3dllm_format.json', 'w') as f:
        json.dump(data_3dllm_source, f)
    print(f'3D-LLM data: {len(data_3dllm_source)} data items Processed.')

# --------------------- Embodied Plan ---------------------------------
if not osp.exists(lang_reformat_path / 'embodiedplan_format.json'):
    embodiedplan_source = []
    embodiedplan_data = json.load(open('data/langdata/EmbodiedDialog/embodied_planning_v2.json'))
    embodiedplan_data.update(json.load(open('data/langdata/EmbodiedDialog/embodied_planning_val_v2.json')))
    embodiedplan_data.update(json.load(open('data/langdata/EmbodiedDialog/embodied_planning_gpt4o.json')))
    plan_count = 0
    for scene_id, data in embodiedplan_data.items():
        for plan_task,steps in data["plan"].items():
            if steps == []:
                continue
            plan_count+=1
            d = {}
            d['scene_id'] = scene_id
            d['lang_type'] = 'embodiedplan:' # TODO: Support grounding
            d['question'] = str(plan_task)
            d['answer'] = steps[0]['desc']

            positive_chars = []
            for object_id in steps[0]["pos"]:
                for p in steps[0]["pos"][object_id]:
                    positive_chars.append((p[0], p[1], int(object_id)))
            positive_chars = sorted(positive_chars)
            positive_dict = OrderedDict()
            for p0, p1, obj_id in positive_chars:
                if (p0, p1) not in positive_dict:
                    positive_dict[(p0, p1)] = [obj_id]
                else:
                    positive_dict[(p0, p1)].append(obj_id)
                                
            d['all_phrases_positions'] = []
            d['object_ids'] = []
            for positives, inst_ids in positive_dict.items():
                # Filtered by gt_inst_ids
                inst_ids = list(set(inst_ids) & set( scene_filter_inst_ids[scene_id]))
                if len(inst_ids) == 0: continue
                d['all_phrases_positions'].append([positives[0],positives[1]])
                d['object_ids'].append(inst_ids)

            embodiedplan_source.append(d)
    with open(lang_reformat_path / 'embodiedplan_format.json', 'w') as f:
        json.dump(embodiedplan_source, f)
    print(f'Embodied Plan: {len(embodiedplan_source)} data items Processed.')

# ------------------------ Embodied Dialog -----------------------------
if not osp.exists(lang_reformat_path / 'embodieddialog_format.json'):
    embodieddialog_source = []
    embodieddialog_data = json.load(open('data/langdata/EmbodiedDialog/embodied_dialog_v2.json'))+\
        json.load(open("data/langdata/EmbodiedDialog/embodied_dialog_val_v2.json"))+\
        json.load(open("data/langdata/EmbodiedDialog/embodied_dialog_gpt4o.json"))

    for d in embodieddialog_data:
        scene_id = d['scene_id']
        d['lang_type'] = 'embodieddialog:' # TODO: Support grounding
        d['question'] = d['history_with_question']
        d['answer'] = d['anwser']
        d.pop('anwser', None)
        d.pop('history_with_question', None)

        positive_chars = []
        for object_id in d["answer_positive"]:
            for p in d["answer_positive"][object_id]:
                positive_chars.append((p[0], p[1], int(object_id)))
        positive_chars = sorted(positive_chars)
        positive_dict = OrderedDict()
        for p0, p1, obj_id in positive_chars:
            if (p0, p1) not in positive_dict:
                positive_dict[(p0, p1)] = [obj_id]
            else:
                positive_dict[(p0, p1)].append(obj_id)
                            
        d['all_phrases_positions'] = []
        d['object_ids'] = []
        for positives, inst_ids in positive_dict.items():
            # Filtered by gt_inst_ids
            inst_ids = list(set(inst_ids) & set( scene_filter_inst_ids[d['scene_id']]))
            if len(inst_ids) == 0: continue
            d['all_phrases_positions'].append(positives)
            d['object_ids'].append(inst_ids)


        positive_chars = []
        for object_id in d["history_with_question_positive"]:
            for p in d["history_with_question_positive"][object_id]:
                positive_chars.append((p[0], p[1], int(object_id)))
        positive_chars = sorted(positive_chars)
        positive_dict = OrderedDict()
        for p0, p1, obj_id in positive_chars:
            if (p0, p1) not in positive_dict:
                positive_dict[(p0, p1)] = [obj_id]
            else:
                positive_dict[(p0, p1)].append(obj_id)
                            
        d['all_phrases_positions_question'] = []
        d['object_ids_question'] = []
        for positives, inst_ids in positive_dict.items():
            # Filtered by gt_inst_ids
            inst_ids = list(set(inst_ids) & set( scene_filter_inst_ids[d['scene_id']]))
            if len(inst_ids) == 0: continue
            d['all_phrases_positions_question'].append(positives)
            d['object_ids_question'].append(inst_ids)

        embodieddialog_source.append(d)
    with open(lang_reformat_path / 'embodieddialog_format.json', 'w') as f:
        json.dump(embodieddialog_source, f)
    print(f'Embodied Dialog: {len(embodieddialog_source)} data items Processed.')

if not osp.exists(lang_reformat_path / 'global_scene_cap.json'):
    scene_lang_source = json.load(open('data/langdata/scene_caption/scene_caption_75.json'))+\
        json.load(open('data/langdata/scene_caption/scene_caption_from_object_70.json'))
    filtered_scene_lang_source = []

    for i, lang in enumerate(scene_lang_source):
        valid_flag = True
        positive_chars = []
        for object_ids in lang["caption_positive"]:
            for p in lang["caption_positive"][object_ids]:
                positive_chars.append((p[0], p[1], int(object_ids)))
        positive_chars = sorted(positive_chars)
        positive_dict = OrderedDict()
        for p0, p1, obj_id in positive_chars:
            if (p0, p1) not in positive_dict:
                positive_dict[(p0, p1)] = [obj_id]
            else:
                positive_dict[(p0, p1)].append(obj_id)
        
        lang['all_phrases_positions_answer'] = []
        lang['object_ids_answer'] = []
        for positives, inst_ids in positive_dict.items():
            if not set(inst_ids).issubset(set( scene_filter_inst_ids[lang['scene_id']])):
                valid_flag = False
                if DEBUG:
                    print(f'{i}: {lang["scene_id"]} uses instance_ids outside the annotation!')
            # Filtered by gt_inst_ids
            inst_ids = list(set(inst_ids) & set( scene_filter_inst_ids[lang['scene_id']]))
            if len(inst_ids) == 0: continue
            lang['all_phrases_positions_answer'].append(positives)
            lang['object_ids_answer'].append(inst_ids)
        
        lang['lang_type'] = 'globalscenecap:'

        # remove the intermediate None type
        if not lang["caption_positive"]:
            continue
        all_phrases_positions = []
        for j, jpp in enumerate(lang["caption_positive"]): # remove intermediate None
            if jpp:
                all_phrases_positions.append(jpp)
        lang['all_phrases_positions'] = all_phrases_positions
        lang['caption'] = replace_punctuation(lang['caption'])
        
        if valid_flag:
            lang.pop("caption_positive")
            filtered_scene_lang_source.append(lang)

    with open(lang_reformat_path / 'global_scene_cap_format.json', 'w') as f:
        json.dump(filtered_scene_lang_source, f)

    print(f'GlobalSceneCaption: {len(filtered_scene_lang_source)} data items Processed.')

