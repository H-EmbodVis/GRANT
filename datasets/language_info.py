import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any
import json
import random
from copy import deepcopy
from datasets.utils import concatenate_texts_with_separator

# load language templates
with open("models/LLM/lan_template.json", 'r') as f:
    lang_template = json.load(f)


class lang_info_data():
    def __init__(self, question=None, answer=None, lang_type=None, positives_question=[], inst_ids_question=[], query_ids_question=None,
                 positives_answer=[], inst_ids_answer=[], query_ids_answer=None, thinking=None, 
                 subtasks=None, actions=None, min_time=None, min_time_action_list=None):

        self.question = question
        self.answer = answer
        self.lang_type = lang_type

        self.positives_question = positives_question
        self.inst_ids_question = inst_ids_question
        self.query_ids_question = query_ids_question

        self.positives_answer = positives_answer
        self.inst_ids_answer = inst_ids_answer
        self.query_ids_answer = query_ids_answer

        self.thinking = thinking
        self.subtasks = subtasks
        self.actions = actions
        self.min_time = min_time
        self.min_time_action_list = min_time_action_list

        self.assert_ListOfList(positives_question)
        self.assert_ListOfList(inst_ids_question)
        self.assert_ListOfList(query_ids_question)
        self.assert_ListOfList(positives_answer)
        self.assert_ListOfList(inst_ids_answer)
        self.assert_ListOfList(query_ids_answer)

        assert self.lang_type.split(':')[0] in ['detection', 'scanrefer', 'm3dref', 'groundedscenecaption', 'scanqa', 'objdesc', 'scenedesc',
                                                'scan2cap', '3dllm', 'alpaca', 'embodieddialog', 'embodiedplan', "globalscenecap", "taskrefer", "taskscheduling"]
        assert self.lang_type.split(':')[-1] in ['text_only', 'with_grounding']
        assert self.lang_type.endswith('with_grounding') or self.lang_type.endswith(
            'text_only'), f'{self.lang_type} has not withgrounding or text_only'
        
        

        self.max_gt_iou = np.nan

    @classmethod
    def from_grounding(cls,
                       raw_text,
                       lang_type,
                       lang_token_inst_id_pair,
                       map_target_to_query,
                       valid_target,
                       support_counting=False, count_instance=True):

        map_num_to_words = lang_template["numbers_to_words"]

        if lang_type.startswith('detection'):
            det = (random.choice(lang_template["detection"]["questions"]),
                   random.choice(lang_template["detection"]["answers_w_o_s"]),
                   random.choice(lang_template["detection"]["answers_wo_o"]),
                   random.choice(lang_template["detection"]["answers_w_o_m"]),
                   random.choice(
                       lang_template["detection"]["counting_problem"]),
                   )
            # TODO: add num to output to improve counting task
            if support_counting:
                gt_inst_ids = np.unique(
                    [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
                probability_of_counting = random.random()
                if probability_of_counting < 0.8:  # counting problem
                    input_text = det[4].format(category=raw_text)
                else:  # det only
                    input_text = det[0].format(category=raw_text)
                if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                    gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                    )
                    if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                        prepare_lan_text = raw_text
                        if count_instance:
                            if probability_of_counting < 0.8:  # counting problem
                                if len(valid_target[gt_inst_ids]) <= 20:
                                    prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                                else:  # use digits
                                    prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                            else:  # det only
                                pass  # Use original text
                        positive = det[3].find("{")
                        positive = (positive, positive+len(prepare_lan_text)+1)
                        output_text = det[3].format(category=prepare_lan_text)
                    else:  # single object
                        positive = det[1].find("{")
                        positive = (positive, positive+len(raw_text))
                        output_text = det[1].format(category=raw_text)
                else:
                    positive = []
                    gt_queries_id = []
                    output_text = det[2].format(category=raw_text)
            else:
                gt_inst_ids = np.unique(
                    [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
                input_text = det[0].format(category=raw_text)
                if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                    gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                    )
                    if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                        prepare_lan_text = raw_text
                        if count_instance:
                            # manually filter: "Several {category}s have been identified in this indoor setting.",
                            if random.random() < 0.5 and not det[3].startswith("Several "):
                                if random.random() < 0.5 and len(valid_target[gt_inst_ids]) <= 20:
                                    prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                                else:  # use digits
                                    prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                            else:
                                pass  # Use original text
                        positive = det[3].find("{")
                        positive = (positive, positive+len(prepare_lan_text)+1)
                        output_text = det[3].format(category=prepare_lan_text)
                    else:  # single object
                        positive = det[1].find("{")
                        positive = (positive, positive+len(raw_text))
                        output_text = det[1].format(category=raw_text)
                else:
                    positive = []
                    gt_queries_id = []
                    output_text = det[2].format(category=raw_text)
        elif lang_type.startswith('scanrefer'):
            grd = (random.choice(lang_template["grounding"]["questions"]),
                   random.choice(lang_template["grounding"]["answers_w_o_s"]),
                   random.choice(lang_template["grounding"]["answers_wo_o"]))
            gt_inst_ids = np.unique(
                [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
            input_text = grd[0].format(grounding_text=raw_text)
            if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                )
                positive = grd[1].find("{")
                positive = (positive, positive+len("object"))
                output_text = grd[1].format(category="object")
            else:
                positive = []
                gt_queries_id = []
                output_text = grd[2].format(category="object")
        elif lang_type.startswith('taskrefer'):
            grd = (random.choice(lang_template["grounding"]["questions"]),
                   random.choice(lang_template["grounding"]["answers_w_o_s"]),
                   random.choice(lang_template["grounding"]["answers_wo_o"]))
            gt_inst_ids = np.unique(
                [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
            input_text = grd[0].format(grounding_text=raw_text)
            if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                )
                positive = grd[1].find("{")
                positive = (positive, positive+len("object"))
                output_text = grd[1].format(category="object")
            else:
                positive = []
                gt_queries_id = []
                output_text = grd[2].format(category="object")
        elif lang_type.startswith('m3dref'):
            multi_grd = (random.choice(lang_template["multi_grounding"]["questions"]),
                         random.choice(
                             lang_template["multi_grounding"]["answers_w_o_s"]),
                         random.choice(
                             lang_template["multi_grounding"]["answers_wo_o"]),
                         random.choice(lang_template["multi_grounding"]["answers_w_o_m"]))
            gt_inst_ids = np.unique(
                [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
            input_text = multi_grd[0].format(grounding_text=raw_text)
            if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                )
                if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                    prepare_lan_text = "object"
                    if count_instance:
                        if random.random() < 0.5:
                            if random.random() < 0.5 and len(valid_target[gt_inst_ids]) <= 20:
                                prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                            else:  # use digits
                                prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                        else:
                            pass  # Use original text
                    positive = multi_grd[3].find("{")
                    positive = (positive, positive+len(prepare_lan_text)+1)
                    output_text = multi_grd[3].format(
                        category=prepare_lan_text)
                else:  # single object
                    positive = multi_grd[1].find("{")
                    positive = (positive, positive+len("object"))
                    output_text = multi_grd[1].format(category="object")
            else:
                positive = []
                gt_queries_id = []
                output_text = multi_grd[2].format(category="object")
        else:
            print(str(self))
            pass

        return cls(
            question=input_text,
            answer=output_text,
            lang_type=lang_type,
            positives_answer=[positive],
            inst_ids_answer=[gt_inst_ids.tolist()],
            query_ids_answer=[gt_queries_id])

    @classmethod
    def from_instruction_following(cls,
                                   instruction_item,
                                   train_mode=False,
                                   ):
        instruction_item = deepcopy(instruction_item)
        if instruction_item['lang_type'].startswith('scanqa'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only'
            )
        elif instruction_item['lang_type'].startswith('objdesc'):
            question = random.choice(lang_template['object_description'])
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_question=[[instruction_item['object_ids'][0][0]]],
                positives_question=[
                    [question.index('object'), question.index('object') + len('object')]]
            )
        elif instruction_item['lang_type'].startswith('scan2cap'):
            question = random.choice(lang_template['scan2cap'])
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_question=[[instruction_item['object_ids'][0][0]]],
                positives_question=[
                    [question.index('object'), question.index('object') + len('object')]],
            )
        elif instruction_item['lang_type'].startswith('scenedesc'):
            question = random.choice(lang_template['scene_description'])
            # flatten inst ids
            unique_object_ids = np.unique(
                [j for i in instruction_item['object_ids'] for j in i]).tolist()
            np.random.shuffle(unique_object_ids)

            if train_mode and np.random.rand() < 0.3:
                lang_type = instruction_item['lang_type'] + ':text_only'
            else:
                lang_type = instruction_item['lang_type'] + ':with_grounding'
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=lang_type,
                inst_ids_question=[unique_object_ids],
                positives_question=[
                    [question.index('objects'), question.index('objects') + len('objects')]],
                inst_ids_answer=instruction_item['object_ids'],
                positives_answer=instruction_item['all_phrases_positions']
            )
        elif instruction_item['lang_type'].startswith('3dllm'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answers'],
                lang_type=instruction_item['lang_type'] + ':text_only'
            )
        elif instruction_item['lang_type'].startswith('embodieddialog'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':with_grounding',
                inst_ids_question=instruction_item['object_ids_question'],
                inst_ids_answer=instruction_item['object_ids'],
                positives_question=instruction_item['all_phrases_positions_question'],
                positives_answer=instruction_item['all_phrases_positions'],
            )
        elif instruction_item['lang_type'].startswith('embodiedplan'):
            plan_question = instruction_item['question']
            plan_answer = instruction_item['answer']
            plan_question = f"{random.choice(lang_template['planning']['requires']).format(a_high_level_task=str(plan_question).lower())}"
            plan_prefix = f"{random.choice(lang_template['planning']['plan_start'])}\n"
            plan_answer = plan_prefix+plan_answer + \
                f"{random.choice(lang_template['planning']['plan_complete'])}"

            return cls(
                question=plan_question,
                answer=plan_answer,
                lang_type=instruction_item['lang_type'] + ':with_grounding',
                inst_ids_answer=instruction_item['object_ids'],
                positives_answer=[[positive[0]+len(plan_prefix), positive[1]+len(
                    plan_prefix)] for positive in instruction_item['all_phrases_positions']],
            )
        elif instruction_item['lang_type'].startswith('alpaca'):
            return cls(
                question=instruction_item["instruction"] +
                instruction_item["input"],
                answer=instruction_item['output'],
                lang_type=instruction_item['lang_type'] + ":text_only",
            )
        elif instruction_item['lang_type'].startswith('globalscenecap'):
            return cls(
                question=random.choice(lang_template["globalscenecap"]),
                answer=instruction_item["caption"],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_answer=instruction_item["object_ids_answer"],
                positives_answer=instruction_item["all_phrases_positions_answer"],
            )
        elif instruction_item['lang_type'].startswith('taskscheduling'):
            question = instruction_item['question']
            answer = instruction_item['answer']
            thinking = instruction_item['thinking'] 
            subtasks = instruction_item['subtasks']
            actions = instruction_item['actions']
            min_time = instruction_item['min_time']
            min_time_action_list = instruction_item['min_time_action_list']

            return cls(
                question=question,
                answer=answer,
                lang_type=instruction_item['lang_type'] + ':with_grounding',
                inst_ids_answer=instruction_item['object_ids'],
                positives_answer=instruction_item['all_phrases_positions'], 
                thinking=thinking,
                subtasks=subtasks,
                actions=actions,
                min_time=min_time,
                min_time_action_list=min_time_action_list,
            )
        else:
            raise NotImplementedError

    def append_prompt_postfix(self):
        # short answer prompt
        if "scanqa" in self.lang_type:
            self.question = self.question + " Please answer with a single word or phrase."

        # with grounding prompt
        if 'with_grounding' in self.lang_type:
            self.question = self.question + ' (with grounding)'

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def set_max_gt_iou(self, max_gt_iou):
        if 'scan2cap' in self.lang_type or 'objdesc' in self.lang_type:
            assert len(self.inst_ids_question) == 1 and len(
                self.inst_ids_question[0]) == 1
            self.max_gt_iou = max_gt_iou[self.inst_ids_question[0][0]].item()

    def set_context_features(self, query_hidden_feature, query_normalized_embed):
        self.query_hidden_feature = query_hidden_feature
        self.query_normalized_embed = query_normalized_embed

    def assert_ListOfList(self, x):
        if not x:
            return
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], (list, tuple)):
                return

        print('Assertion Error ListofList')

    def remap_inst_ids(self, mapping):
        def inplace_replace_insts(rawtext_posinsts_tmp, instance_mapping_tmp):
            for i, posinsts in enumerate(rawtext_posinsts_tmp):
                for j in range(len(posinsts)): 
                    if rawtext_posinsts_tmp[i][j] in instance_mapping_tmp:
                        rawtext_posinsts_tmp[i][j] = instance_mapping_tmp[rawtext_posinsts_tmp[i][j]]
                    else:
                        rawtext_posinsts_tmp[i][j] = 0
        inplace_replace_insts(self.inst_ids_question, mapping)
        inplace_replace_insts(self.inst_ids_answer, mapping) 

    def __str__(self):
        return (f'lang_info_data >>> \n') +\
            (f'question: {self.question}\n') +\
            (f'answer: {self.answer}\n') +\
            (f'lang_type: {self.lang_type}\n') +\
            (f'positives_question: {self.positives_question}\n') + \
            (f'inst_ids_question: {self.inst_ids_question}\n') + \
            (f'query_ids_question: {self.query_ids_question}\n') + \
            (f'positives_answer: {self.positives_answer}\n') + \
            (f'inst_ids_answer: {self.inst_ids_answer}\n') + \
            (f'query_ids_answer: {self.query_ids_answer}\n')


class grounding_data:
    def __init__(self):
        self.texts = [] 
        self.types = []
        self.positives = []
        self.gt_insts = []

        self.concat_texts, self.concat_positives, self.concat_gt_insts, self.concat_types = [], [], [], []

    def add_detection(self, class_label, gt_insts):
        self.texts.append(class_label + '.')
        self.gt_insts.append([gt_insts])
        self.positives.append([[0, len(class_label)]])
        self.types.append('detection:with_grounding') 

    def add_grounding(self, grounding_text, gt_insts, positives, grounding_type):
        self.texts.append(deepcopy(grounding_text))
        self.gt_insts.append(deepcopy(gt_insts))
        self.positives.append(deepcopy(positives))
        self.types.append(deepcopy(grounding_type)+':with_grounding')

    def shuffle_grounding(self):
        random_text_indices = np.arange(
            len([typ for typ in self.types if not typ.startswith('detection')]))
        np.random.shuffle(random_text_indices)

        self.texts = np.asarray(self.texts, dtype=object)
        self.types = np.asarray(self.types, dtype=object)
        self.positives = np.asarray(self.positives, dtype=object)
        self.gt_insts = np.asarray(self.gt_insts, dtype=object)

        withdet_indices = np.asarray(np.arange(len(self.texts) - len(random_text_indices)).tolist(
        ) + (len(self.texts) - len(random_text_indices) + random_text_indices).tolist())
        self.texts = self.texts[withdet_indices].tolist()
        self.types = self.types[withdet_indices].tolist()
        self.positives = self.positives[withdet_indices].tolist()
        self.gt_insts = self.gt_insts[withdet_indices].tolist()

    def concat_multi_grounding(self, tokenizer, max_batch_tokens, max_tokens, num_concat_texts):
        self.concat_texts, self.concat_positives, self.concat_gt_insts, self.concat_types = concatenate_texts_with_separator(
            tokenizer, self.texts, max_batch_tokens, max_tokens=max_tokens,
            num_concat_texts=num_concat_texts, raw_texts_poschars=self.positives, raw_texts_posinsts=self.gt_insts,
            raw_texts_type=self.types, shuffle=False, text_separator=' ', concat=True)

    def remap_inst_ids(self, mapping):
        instances_to_delete = set()
        self.concat_texts = np.stack(self.concat_texts)
        for gt_insts in self.concat_gt_insts:
            for i, posinsts in enumerate(gt_insts):
                if isinstance(posinsts, (list, np.ndarray)):
                    for j in range(len(posinsts)):
                        if posinsts[j] in mapping:
                            gt_insts[i][j] = mapping[gt_insts[i][j]]
                        else:
                            instances_to_delete.add(posinsts[j])
                else:
                    assert False
        for inst_id in instances_to_delete:
            for i, gt_insts in enumerate(self.concat_gt_insts):
                for j, posinsts in enumerate(gt_insts):
                    if inst_id in posinsts:
                        self.concat_gt_insts[i][j].remove(inst_id)

    def compute_positive_maps(self, tokenizer):
        pass
