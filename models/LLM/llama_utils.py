import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Any
from dataclasses import dataclass, field
import re
@dataclass
class MiniBatchData:
    batch:       Optional[Any] = field(default=None)

class grounded_3d_llm_data:
    def __init__(self,
                 input_text: Optional[str] = None,
                 input_ids: Optional[List[int]] = None,
                 output_text: Optional[str] = None,
                 output_ids: Optional[List[int]] = None,
                 answer_gt_query_ids: Optional[List[List[int]]] = None,
                 answer_grounding_token_pos: Optional[List[int]] = None,
                 question_gt_query_ids: Optional[List[List[int]]] = None,
                 question_grounding_token_pos: Optional[List[int]] = None,
                 gt_instance_predicted_iou: Optional[float] = None,
                 ref_token_mask: Optional[List[int]] = None,
                 instance_feature: Optional[Any] = None,
                 instance_embed: Optional[Any] = None,
                 use_input_referent: Optional[bool] = None,
                 input_referent_mask: Optional[List[int]] = None,
                 input_referent: Optional[Any] = None,
                 grouped_indices: Optional[List[dict]] = None,
                 eval_type: Optional[str] = None,
                 output_think_texts: Optional[str] = None,
                 min_time: Optional[Any] = None,
                 min_time_action_list: Optional[Any] = None,
                 time_token_mask: Optional[List[int]] = None,
                 ):
                 
        '''
            we only check if answer_grounding_token_pos and question_grounding_token_pos are empty
            if they are empty, corresponding ids will not be saved
            both answer_grounding_token_pos and question_grounding_token_pos will be formated as:
                                [[start_idx, end_idx, number_of_id],... ]
        '''

        if 'with_grounding' in eval_type and answer_grounding_token_pos != [] and answer_grounding_token_pos is not None:
            answer_grounding_token_pos, answer_gt_query_ids = self.zip_interval(
                answer_grounding_token_pos, answer_gt_query_ids)
        else:
            answer_grounding_token_pos, answer_gt_query_ids = None, None
        if question_grounding_token_pos != [] and question_grounding_token_pos is not None:
            question_grounding_token_pos, question_gt_query_ids = self.zip_interval(
                question_grounding_token_pos, question_gt_query_ids)
        else:
            question_grounding_token_pos, question_gt_query_ids = None, None

        self.input_text = input_text
        self.input_ids = input_ids
        self.output_text = output_text
        self.output_ids = output_ids
        self.answer_gt_query_ids = answer_gt_query_ids
        self.answer_grounding_token_pos = answer_grounding_token_pos
        self.question_gt_query_ids = question_gt_query_ids
        self.question_grounding_token_pos = question_grounding_token_pos
        self.gt_instance_predicted_iou = gt_instance_predicted_iou
        self.ref_token_mask = ref_token_mask
        self.time_token_mask = time_token_mask
        self.instance_feature = instance_feature
        self.instance_embed = instance_embed
        self.input_referent_mask = input_referent_mask
        self.input_referent = input_referent
        self.grouped_indices = grouped_indices
        self.eval_type = eval_type
        self.output_think_texts = output_think_texts
        self.min_time = min_time
        self.min_time_action_list = min_time_action_list

        self.datasets_use_input_referent = [
            "embodieddialog", "objdesc", "scan2cap", "scenedesc"]

        self._use_input_referent = use_input_referent

    @property
    def use_input_referent(self):
        '''
        Usage: auto detect or manually set whether to use input referent. You can always set use_input_referent to be true.
        '''
        if self._use_input_referent is None:
            if any([i in self.eval_type for i in self.datasets_use_input_referent]):
                self._use_input_referent = True
            else:
                self._use_input_referent = False
        return self._use_input_referent

    @use_input_referent.setter
    def use_viusal_prompt(self, use_input_referent=None):
        assert isinstance(use_input_referent, bool)
        self._use_input_referent = use_input_referent

    @staticmethod
    def zip_interval(grounding_token_pos,
                     gt_instance_ids):
        grounding_token_pos_with_counter = []
        new_gt_instance_ids = []
        for raw_token_pos, id in zip(grounding_token_pos, gt_instance_ids):
            if raw_token_pos and id:
                grounding_token_pos_with_counter.append(
                    [raw_token_pos[0], raw_token_pos[1], len(id)])
                new_gt_instance_ids.append(id)
        return grounding_token_pos_with_counter, new_gt_instance_ids

    @property
    def output_range(self):
        r"""
        Since we many add separate token between modalities,and the position of output feature may shift. This property stores the index of output text.
        """
        s = self.instance_feature.shape[0] + self.input_ids.shape[0]
        s += 1 
        e = s + self.ref_token_mask.shape[0]
        return s, e
    
    @property
    def get_time_token_output_range(self):
        r"""
        Since we many add separate token between modalities,and the position of output feature may shift. This property stores the index of output text.
        """
        s = self.instance_feature.shape[0] + self.input_ids.shape[0]
        s += 1  
        e = s + self.time_token_mask.shape[0] 
        return s, e

    def group_true_indices(self,
                           output_last_hidden_state,
                           MLP,
                           ref_token_id):
        '''
        Usage: convert corresponding ref tokens from llm hidden states to instacne embeddings
        input:
            instance: grounded_3d_llm_data : get gt query id
            output_last_hidden_state: llm last hidden states
            label: double check corresponding token id == ref token
        return:
            a dict stores all information about grounded phrases
        '''
        boolean_list = self.ref_token_mask 
        label = self.output_ids
        assert boolean_list.shape[0] == output_last_hidden_state.shape[0]
        groups = {}
        current_group = None
        for idx, value in enumerate(boolean_list):
            if value:
                if current_group is None:
                    current_group = len(groups) + 1
                    groups[str(current_group)] = [idx]
                else:
                    groups[str(current_group)].append(idx)
            else:
                current_group = None
        for idx, key in enumerate(groups):
            assert torch.all(label[groups[key]] == ref_token_id)
            groups[key] = [i-1 for i in groups[key]]
            groups[key] = {"index": groups[key],
                           "last_hidden_state": output_last_hidden_state[groups[key], :],
                           "labels": label[torch.tensor(groups[key])+1],
                           "gt": self.answer_gt_query_ids[idx],
                           "query": MLP(output_last_hidden_state[groups[key], :]).float(),
                           }
        return groups

    @staticmethod
    def build_input_from_segments(tokenizer,  
                                  input_text, 
                                  prompts, 
                                  use_system_prompt=True,
                                  output_text=None,
                                  use_input_referent=False, 
                                  inference=False,
                                  use_single_ref_token=True,
                                  question_grounding_token_pos=None,
                                  answer_grounding_token_pos=None,
                                  eval_type = None,
                                  think=False, 
                                  think_text=None,
                                  reason_type=None,
                                  ):
        gs_token_id = tokenizer.gs_token_id
        ge_token_id = tokenizer.ge_token_id
        eos_token_id = tokenizer.eos_token_id
        ref_token_id = tokenizer.ref_token_id
        inref_token_id = tokenizer.inref_token_id 

        instance = {
            "input_ids": None,
            "input_referent_mask": None,
            "lm_labels": None,
            "ref_token_mask": None,
            "time_token_mask": None
        }
        def convert_text_to_ids(text): return tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(text))
        rules = prompts["rules"]
        assistant_prefix = rules["assistant"]["ids"]
        user_prefix = rules["user"]["ids"]
        if use_system_prompt:
            user_prefix = torch.cat(
                (prompts["system_prompt"]["ids"], user_prefix))
        if use_input_referent:
            input_text_add_input_referent_ids = []
            last_end = 0
            if question_grounding_token_pos and question_grounding_token_pos[0]:
                for start, end, count in question_grounding_token_pos:
                    p = np.random.rand()
                    if p < 1.0/3 or inference:  
                        input_text_add_input_referent_ids += convert_text_to_ids(
                            input_text[last_end:start]) + [inref_token_id]*(count+2)  
                    elif 2.0/3 > p > 1.0/3:
                        input_text_add_input_referent_ids += convert_text_to_ids(
                            input_text[last_end:end]) + [inref_token_id]*(count+2) 
                    else:
                        input_text_add_input_referent_ids += convert_text_to_ids(input_text[last_end:start]) + [inref_token_id]*(
                            count+2) + convert_text_to_ids(input_text[start:end])  
                    last_end = end
                input_text_add_input_referent_ids += convert_text_to_ids(
                    input_text[last_end:])+[eos_token_id]  
            else:
                input_text_add_input_referent_ids = convert_text_to_ids(
                    input_text)+[eos_token_id]
            instance['input_ids'] = torch.tensor(
                input_text_add_input_referent_ids, dtype=int)
            input_ids = instance['input_ids'].clone(
            ).detach().to(dtype=torch.int)
            instance["input_referent_mask"] = (
                (input_ids == inref_token_id)).to(dtype=bool)
        else:
            instance['input_ids'] = torch.tensor(
                convert_text_to_ids(input_text)+[eos_token_id], dtype=int)
            instance["input_referent_mask"] = torch.zeros(
                instance['input_ids'].shape[0], dtype=bool)
        instance["input_ids"] = torch.cat((user_prefix, instance["input_ids"]))


        instance["input_referent_mask"] = torch.cat(
            (torch.zeros(user_prefix.shape[0], dtype=bool), instance["input_referent_mask"]))
        if inference: 
            return instance["input_ids"], instance["input_referent_mask"]
        
        output_text_add_ref_ids = []
        last_end = 0
        if answer_grounding_token_pos and answer_grounding_token_pos[0]:
            for start, end, count in answer_grounding_token_pos:
                if use_single_ref_token:
                    count = 1
                output_text_add_ref_ids += convert_text_to_ids(output_text[last_end:start]) + [
                    gs_token_id] + convert_text_to_ids(output_text[start:end]) + [ref_token_id]*count + [ge_token_id] 
                last_end = end
            output_text_add_ref_ids += convert_text_to_ids(
                output_text[last_end:])

        else:
            output_text_add_ref_ids += convert_text_to_ids(output_text)

        output_text_add_ref_ids += [eos_token_id]

        instance["lm_labels"] = torch.tensor(output_text_add_ref_ids, dtype=int)
        instance['time_token_mask'] = None
        instance['plan_token_mask'] = None 

        if not think: 
            instance["lm_labels"] = torch.cat(
                (assistant_prefix, instance["lm_labels"]))
            
        elif think and think_text != None and "taskscheduling" in eval_type: 
            output_think_add_think_ids = convert_text_to_ids(think_text)
            instance_think_ids = torch.tensor(
            output_think_add_think_ids, dtype=int)
            instance["lm_labels"] = torch.cat(
                (assistant_prefix, instance_think_ids, instance["lm_labels"]))

        
        instance['ref_token_mask'] = (
            instance["lm_labels"].clone().detach() == ref_token_id).to(dtype=bool)


        return instance["input_ids"], instance["lm_labels"], instance["ref_token_mask"], instance["input_referent_mask"], instance['time_token_mask']


def get_loss_for_each_type(output, batch):
    if hasattr(output, 'loss_before_mean') and output.loss_before_mean is not None:
        loss_data_type = {}
        data_type_counter = {}
        all_data_eval_types = np.asarray(
            [i.eval_type.split(':')[0] for i in batch]) 
        for k in np.unique(all_data_eval_types):
            specific_type = all_data_eval_types == k
            loss_data_type[f'lm_loss_{k}_detach'] = output.loss_before_mean[specific_type].mean(
            )
            data_type_counter[k] = specific_type.sum()
    else:
        loss_data_type = {}
    return loss_data_type


def extract_decoder_hidden_states(
    generate_output_dict,
    hidden_layer_idx=-1,
):
    """
    Extracts the decoder hidden states representation from
    GreedySearchEncoderDecoderOutput and BeamSearchEncoderDecoderOutput,
    associated with the `sequences` output.
    - generate_output_dict: output dict from the model.generate() method
      you should add the following arguments to generate:
        - output_hidden_states=True
        - output_scores=True
        - return_dict_in_generate=True
    - hidden_layer_idx: index of the layer to extract the representation from (-1 == last one)
    """
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, \
        BeamSearchDecoderOnlyOutput, \
        GreedySearchEncoderDecoderOutput, \
        BeamSearchEncoderDecoderOutput, \
        BeamSampleDecoderOnlyOutput

    greedy = any([isinstance(generate_output_dict, i) for i in [GreedySearchDecoderOnlyOutput,
                                                                GreedySearchEncoderDecoderOutput]])
    beamy = any([isinstance(generate_output_dict, i) for i in [BeamSearchDecoderOnlyOutput,
                                                               BeamSearchEncoderDecoderOutput,
                                                               BeamSampleDecoderOnlyOutput]])

    if greedy:
        # in greedy decoding, the beam_indices is not present, so we create one
        # where the first beam is always selected
        scores = generate_output_dict['scores']
        device = generate_output_dict['sequences'].device
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1)
        beam_indices = beam_indices.expand(-1, len(scores)).to(device)
    elif beamy: # 默认是beam
        if 'beam_indices' not in generate_output_dict:
            raise RuntimeError(
                "You should export the scores with output_scores=True when "
                "calling extract_decoder_hidden_states with "
                "BeamSearchEncoderDecoderOutput"
            )
        beam_indices = generate_output_dict['beam_indices'].clone()
    else:
        raise NotImplementedError(
            "extract_decoder_hidden_states only works with "
            "GreedySearch...Output and BeamSearch...Output "
            "output types."
        )
    # handling of the target length and preparing the masking for tokens
    # outside of that length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]
    beam_indices[beam_indices_mask] = 0
    seqlen = generate_output_dict['sequences'].shape[1] - 1
    # creating the output hidden_states representation in format:
    # [bsz * beam_width ; seqlen ; featdim]
    if "Encoder" in str(type(generate_output_dict)):
        decoder_hidden_states = torch.stack([
            generate_output_dict['decoder_hidden_states'][i][hidden_layer_idx][:, 0, :].index_select(
                dim=0,
                index=beam_indices[:, i]  # reordering using the beam_indices
            )
            for i in range(seqlen)
        ]).transpose(0, 1)
    else:
        decoder_hidden_states = [
            generate_output_dict['hidden_states'][i][hidden_layer_idx][:, 0, :].index_select(
                dim=0,
                index=beam_indices[:, i]  # reordering using the beam_indices
            )
            for i in range(seqlen)
        ]
        decoder_hidden_states = torch.stack([torch.zeros(decoder_hidden_states[0].shape).to(
            "cuda")] + decoder_hidden_states).transpose(0, 1)
    return decoder_hidden_states

