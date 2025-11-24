import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
from typing import Optional, List, Any
import random
from torchvision.ops import sigmoid_focal_loss
from datasets.language_info import lang_info_data
import gc # Garbage Collection

from transformers import LlamaTokenizer
from peft import LoraConfig, get_peft_model
from models.misc import print_grad_status
import glob

from .modeling_llama import LlamaForCausalLM, LlamaModel
from .llama_utils import * 
from .plan_utils import call_task_planner 

from transformers import LogitsProcessorList, StoppingCriteriaList


def load_llama_model_and_tokenizer(llama_config):
    llama_config.vicuna_weight_path = glob.glob(
        f"{llama_config.root_path}{llama_config.vicuna_version}/*pytorch_model*.bin")

    llama_config.llama_dim = llama_config.hidden_size
    llama_config.tokenizer_path = f"{llama_config.root_path}{llama_config.vicuna_version}/"
    llama_config.model_path = f"{llama_config.root_path}{llama_config.vicuna_version}"

    # prepare llama tokenizer(load & and special tokens)
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_config.tokenizer_path,
                                                     use_fast=False,
                                                     legacy=False)

    llama_tokenizer.add_tokens([llama_config.ref_token,
                                llama_config.gs_token,
                                llama_config.ge_token,
                                llama_config.inref,
                                ], special_tokens=True)
    llama_tokenizer.ref_token_id = llama_tokenizer(
        llama_config.ref_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.gs_token_id = llama_tokenizer(
        llama_config.gs_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.ge_token_id = llama_tokenizer(
        llama_config.ge_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.inref_token_id = llama_tokenizer(
        llama_config.inref, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.ref_token = "<GRU>"
    llama_tokenizer.gs_token = "<p>" 
    llama_tokenizer.ge_token = "</p>"
    llama_tokenizer.inref = "<inref>"  


    if llama_config.think:
        # construct plan tokens
        llama_tokenizer.add_tokens([
                                    llama_config.plan_token,
                                    ], special_tokens=True)
        llama_tokenizer.plan_token_id = llama_tokenizer(
            llama_config.plan_token, add_special_tokens=False)['input_ids'][0] 
        llama_tokenizer.plan_token = "<plan>"



    # init llama model
    llama_model = LLama3dForCausalLM(config=llama_config,
                                     llama_tokenizer=llama_tokenizer,
                                     gradient_checkpointing=True)

    # prepare vicuna weight 
    if llama_config.load_pretrain_weight:
        vicuna_weight = {}
        assert not llama_config.vicuna_weight_path[0].split(
            ".")[-1] == "safetensor", "currently we only support torch.bin file"
        for path in llama_config.vicuna_weight_path:
            weights = torch.load(path, map_location=torch.device('cpu')) 
            vicuna_weight.update(weights)
        print("Loading LLM weights...")
        llama_model.load_state_dict(vicuna_weight, strict=False)

    llama_model.model.wte = llama_model.resize_token_embeddings(
        len(llama_tokenizer))
    # =============== apply lora ===========================

    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                and all(
                    [
                        x not in name
                        for x in [
                            "instance2embed",
                            "hidden_state2query",
                            "hidden_state2reason_space", 
                        ]
                    ]
                )
                and any([x in name for x in lora_target_modules])
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    lora_target_modules = find_linear_layers(
        llama_model, llama_config.lora_target_modules)
    peft_config = LoraConfig(
        r=llama_config.lora_r,
        lora_alpha=llama_config.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=llama_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llama_model = get_peft_model(llama_model, peft_config)
    llama_model.print_trainable_parameters()
    for name, param in llama_model.named_parameters():
        if any(config_item in name for config_item in llama_config.train_layer_list):
            param.requires_grad = True
        else:
            param.requires_grad = False
    llama_model = llama_model.bfloat16()
    if llama_config.use_checkpoint:
        llama_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return llama_model, llama_tokenizer

class LLama3dMetaModel:
    def __init__(
        self,
        config,
        **kwargs,  
    ):
        super(LLama3dMetaModel, self).__init__(config)
        # config & initialize model
        self.initialize_LLama3d_modules(config)
        self.config = config

        # projection layers
        # project instance to llm embed
        self.instance2embed = nn.Sequential(
            nn.Linear(self.instance_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.llama_dim),
        )
        # project last hidden state to instance query
        self.hidden_state2query = nn.Sequential(
            nn.Linear(self.llama_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.instance_dim)
        )
        self.vision_prompt_projection = nn.Sequential(
            nn.Linear(self.instance_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.llama_dim),
        )


    def initialize_LLama3d_modules(self, config):
        self.llama_dim = config.llama_dim
        self.instance_dim = config.instance_dim


class LLama3dModel(LLama3dMetaModel, LlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LLama3dModel, self).__init__(config, **kwargs)


class LLama3dForCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        config,
        sample_rate=1.0,  
        subsample=True,
        llama_tokenizer=None,
        prompts=None,
        t=0.1,
        **kwargs,
    ):
        super().__init__(config)

        # reduce batch size to avoid OOM
        self.sample_rate = sample_rate
        self.subsample = subsample

        # truncation to avoid OOM
        self.do_truncation = config.do_truncation
        self.truncation_length = config.truncation_length

        if llama_tokenizer is not None:
            self.llama_tokenizer = llama_tokenizer
        else:
            raise NotImplementedError

        # temperature for contrastive learning
        self.t = t

        self.use_single_ref_token = config.use_single_ref_token
        self.config = config

        # param for focal loss
        self.eps = 1e-12
        self.gamma = 2.0
        self.alpha = 0.25

        # config special tokens
        
        self.pad_token_id = llama_tokenizer.eos_token_id 
        self.ref_token_id = llama_tokenizer.ref_token_id
        self.gs_token_id = llama_tokenizer.gs_token_id  # grounding start token
        self.ge_token_id = llama_tokenizer.ge_token_id  # grounding end token
        self.eos_token_id = llama_tokenizer.eos_token_id  # end of sentence token



        def convert_text_to_ids(text): return self.llama_tokenizer.convert_tokens_to_ids(
            self.llama_tokenizer.tokenize(text))
        self.model = LLama3dModel(config, **kwargs) 

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        if prompts is not None:
            self.prompts = prompts
        else:
            system_prompt = "<s> SYSTEM: A chat between a curious user and a 3D AI assistant. The assistant gives helpful and polite answers to the user's questions."
            self.prompts = {
                "system_prompt": {
                    "lan": system_prompt,
                    "ids": torch.tensor(convert_text_to_ids(system_prompt))
                },
                "rules": {"user": {
                    "lan": "<s> USER:",
                    "ids": torch.tensor(convert_text_to_ids("<s> USER:"))},
                    "assistant": {
                        "lan": "ASSISTANT:",
                        "ids": torch.tensor(convert_text_to_ids("ASSISTANT:"))}
                },
                "QA_grounding": "",  # postfix
            }
        self.beam_size = config.beam_size
        self.post_init()

    def _merge_input_ids_with_instance_features(self,  batch, inference=False):
        batch_size = len(batch)
        visual_embeddings_list = []
        language_embeddings_list = []
        input_text_list = []
        eos_token_embeds = self.model.embed_tokens(torch.tensor(
            [self.eos_token_id], device=self.device)).to(dtype=torch.bfloat16) 
        for instance in batch: 
            instance: grounded_3d_llm_data 

            visual, lan, label = instance.instance_feature, instance.input_ids, instance.output_ids 
            input_text_list.append(instance.input_text)
            assert visual.shape[
                0] == 100, "if only provide instance queries, feature should have the shape of (100,128)"
            visual_embeddings_list.append(
                self.model.instance2embed(visual.to(dtype=torch.bfloat16)))
            visual_embeddings_list[-1] = torch.cat(
                (visual_embeddings_list[-1], eos_token_embeds), dim=0) 
            language_embeddings_list.append(
                self.model.embed_tokens(lan).to(dtype=torch.bfloat16))
            

            if instance.input_referent_mask.any():
                instance.input_referent = []
                find_inref = 0
                try:
                    for question_quries_id in instance.question_gt_query_ids:
                        # shuffle within one phrase
                        question_quries_id = np.array(question_quries_id)
                        np.random.shuffle(question_quries_id)
                        instance.input_referent.append(self.model.vision_prompt_projection(
                            visual[question_quries_id, :].to(dtype=torch.bfloat16)))
                        while find_inref < instance.input_referent_mask.shape[0] and instance.input_referent_mask[find_inref] != True:
                            find_inref += 1
                        assert find_inref != instance.input_referent_mask.shape[0] - \
                            1, "can not find a corresponding position for insertion"
                        assert instance.input_referent_mask[find_inref+1:find_inref+1+len(question_quries_id)].all(
                        ), "length of interval do not match number of input referent"
                        language_embeddings_list[-1][find_inref+1:find_inref+1 +
                                                     len(question_quries_id)] = instance.input_referent[-1]
                        find_inref += len(question_quries_id)+2
                except Exception as e:
                    print(e)
                    from IPython import embed
                    embed()
                    continue
            if not inference: 
                language_embeddings_list[-1] = torch.cat(
                    (language_embeddings_list[-1], self.model.embed_tokens(label).to(dtype=torch.bfloat16)), dim=0)

        embed_list = []
        max_length_input = 0
        embed_length = []
        for visual, lan in zip(visual_embeddings_list, language_embeddings_list):
            visual_len = len(visual)
            embed_list.append(torch.cat((visual, lan), dim=0)) 
            max_length_input = max(embed_list[-1].shape[0], max_length_input)
            embed_length.append(embed_list[-1].shape[0])

        pad_token_embeds = self.model.embed_tokens(torch.tensor(
            [self.pad_token_id], device=self.device)).to(dtype=torch.bfloat16) 
        
        final_embedding = []
        final_attention_mask = torch.zeros(
            (batch_size, max_length_input), device=self.device, dtype=bool)
        for idx, embed in enumerate(embed_list):
            final_embedding.append(torch.cat(
                (embed, pad_token_embeds.repeat(max_length_input-embed.shape[0], 1)), dim=0))
            final_attention_mask[idx, :embed.shape[0]] = True 
        final_embedding = torch.stack(final_embedding)
        if inference:
            return final_embedding.to(self.device, dtype=torch.bfloat16), final_attention_mask.to(self.device)
        
        label_list = []
        for instance, end in zip(batch, embed_length):
            output_lan = instance.output_ids 
            label = (torch.ones(max_length_input)*-100).type(torch.LongTensor) 
            label[end-output_lan.shape[0]:end] = output_lan 

            label_list.append(label)
        final_labels = torch.stack(label_list)
        assert final_embedding.shape[:2] == final_labels.shape
        return final_embedding.to(self.device, dtype=torch.bfloat16), final_attention_mask.to(self.device), final_labels.to(self.device).type(torch.LongTensor)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs: 
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def calculate_grounding_loss(self, batch, use_single_ref_token, temperature=0.07, loss_type="no"):
        loss = torch.tensor(0.0)
        sim_list = []
        gt_list = []
        for instance in batch: 
            for key in instance.grouped_indices.keys():
                pair = instance.grouped_indices[key]
                pred_ref = pair["query"]
                gt_ids = pair["gt"]
                if use_single_ref_token:  
                    gt_labels = torch.zeros(
                        instance.instance_embed.shape[0], device=self.device).float() 
                    gt_labels[gt_ids] = True
                    
                    features_norm = F.normalize(pred_ref, p=2, dim=1).float() 
                    embeddings_norm = F.normalize(
                        instance.instance_embed, p=2, dim=1).float() 
                    query_sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

                    cosine_sim = torch.matmul(
                        features_norm, embeddings_norm.T)/self.t 
                    sim_list.append(cosine_sim.squeeze())
                    gt_list.append(gt_labels)
                else: 
                    gt_labels = torch.zeros(
                        (pred_ref.shape[0], instance.instance_embed.shape[0]), device=self.device).float()
                    gt_labels[:, gt_ids] = True

                    features_norm = F.normalize(pred_ref, p=2, dim=1).float()
                    embeddings_norm = F.normalize(
                        instance.instance_embed, p=2, dim=1).float()
                    cosine_sim = torch.matmul(
                        features_norm, embeddings_norm.T)/self.t
                    probs = torch.sigmoid(cosine_sim)
                    neg_cost = -(1 - probs + self.eps).log() * \
                        (1 - self.alpha) * probs.pow(self.gamma)
                    pos_cost = -(probs + self.eps).log() * \
                        self.alpha * (1 - probs).pow(self.gamma)
                    cls_cost = torch.einsum(
                        'nc,mc->nm', pos_cost, gt_labels) + torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
                    try:
                        row_ind, col_ind = linear_sum_assignment(
                            cls_cost.cpu().detach().numpy(), maximize=False)
                    except Exception as e:
                        print("calculate_grounding_loss", e, "\n", features_norm, "\n",
                              "embeddings_norm\n", embeddings_norm, '\n', "you may try to use bfloat16")
                        loss = torch.tensor(0.0)
                        return loss
                    matched_loss = cls_cost[row_ind, col_ind].sum()
                    if len(row_ind) > 0:
                        loss = loss + matched_loss/len(row_ind)

        if use_single_ref_token:  
            if sim_list:
                sim_list = torch.stack(sim_list) 
                gt_list = torch.stack(gt_list) 
                loss = sigmoid_focal_loss(
                    sim_list, gt_list, alpha=self.alpha, gamma=self.gamma, reduction='sum')
                loss = loss/(gt_list.sum() + 1) 

                if loss_type == "contrastive":
                    target_query_idx = gt_list.argmax(dim=1)
                    sim_matrix = sim_list / temperature
                    loss_i = F.cross_entropy(sim_matrix, target_query_idx)
                    gt_list_T = gt_list.T
                    query_has_gru_mask = gt_list_T.sum(dim=1) > 0
                    query_indices = torch.nonzero(query_has_gru_mask).squeeze(-1)

                    if len(query_indices) > 0:
                        target_gru_idx = gt_list_T[query_indices].argmax(dim=1)
                        sim_matrix_T = sim_list.T / temperature
                        selected_sim = sim_matrix_T[query_indices]
                        loss_t = F.cross_entropy(selected_sim, target_gru_idx)
                        loss = (loss_i + loss_t) / 2
                    else:
                        loss = loss_i
                
        return loss

    def model_forward(self,
                      batch_input_text_list: list,  
                      batch_output_text_list: list, 
                      batch_instance_queries_hidden_state: list,
                      batch_instance_queries_normalized_embed: list,
                      batch_eval_types: list,
                      batch_gt_inst_ids: list,
                      batch_output_think_texts=None,
                      batch_min_time=None,
                      batch_min_time_action_list=None,
                      **kwargs
                      ):
        batch = [] 
        if self.do_truncation:
            num_of_truncated = 0

        for input_text, output_text, instance_feat, instance_embed, eval_type, gt_instance_ids, output_think_texts, min_time, min_time_action_list in zip(batch_input_text_list,
                                                                                                    batch_output_text_list,
                                                                                                    batch_instance_queries_hidden_state,
                                                                                                    batch_instance_queries_normalized_embed,
                                                                                                    batch_eval_types,
                                                                                                    batch_gt_inst_ids,
                                                                                                    batch_output_think_texts,
                                                                                                    batch_min_time,
                                                                                                    batch_min_time_action_list,
                                                                                                    ):
            gt_instance_ids: lang_info_data
            gt_instance_ids = gt_instance_ids[1]

            answer_grounding_token_pos, answer_gt_query_ids = gt_instance_ids.positives_answer, gt_instance_ids.query_ids_answer
            question_grounding_token_pos, question_gt_query_ids = gt_instance_ids.positives_question, gt_instance_ids.query_ids_question

            instance = grounded_3d_llm_data(input_text=input_text,
                                            output_text=output_text,
                                            instance_embed=instance_embed,
                                            instance_feature=instance_feat,
                                            answer_gt_query_ids=answer_gt_query_ids,
                                            answer_grounding_token_pos=answer_grounding_token_pos,
                                            question_gt_query_ids=question_gt_query_ids,
                                            question_grounding_token_pos=question_grounding_token_pos,
                                            eval_type=eval_type,
                                            output_think_texts=output_think_texts,
                                            min_time=min_time,
                                            min_time_action_list=min_time_action_list,
                                            ) 

            input_ids, lm_labels, ref_token_mask, input_referent_mask, time_token_mask= instance.build_input_from_segments(use_system_prompt=self.config.use_system_prompt,
                                                                                                            tokenizer=self.llama_tokenizer,
                                                                                                            input_text=instance.input_text,
                                                                                                            output_text=instance.output_text,
                                                                                                            prompts=self.prompts,
                                                                                                            use_input_referent=instance.use_input_referent,
                                                                                                            answer_grounding_token_pos=instance.answer_grounding_token_pos,
                                                                                                            question_grounding_token_pos=instance.question_grounding_token_pos,
                                                                                                            use_single_ref_token=self.use_single_ref_token,
                                                                                                            eval_type = instance.eval_type,
                                                                                                            think=self.config.think,
                                                                                                            think_text=output_think_texts,
                                                                                                            reason_type=self.config.reason_type,
                                                                                                            )    


            if self.do_truncation:
                if input_ids.shape[0] + lm_labels.shape[0] + instance_feat.shape[0] > self.truncation_length:
                    left_output_length = self.truncation_length - \
                        input_ids.shape[0] - instance_feat.shape[0]
                    if left_output_length <= 0:
                        print(f"Ignore the long QA: ``{eval_type}''.")
                        continue
                    ref_token_mask = ref_token_mask[:left_output_length]
                    # lm label should not end with seg token
                    for i in range(len(ref_token_mask) - 1, -1, -1):
                        if ref_token_mask[i]:
                            left_output_length -= 1
                        else:
                            break
                    ref_token_mask = ref_token_mask[:left_output_length]
                    lm_labels = lm_labels[:left_output_length]
                    num_of_truncated += 1
                    # print('truncated', eval_type, input_text, output_text)
            instance.output_ids = lm_labels.to(self.device)

            instance.input_ids = input_ids.to(self.device)
            instance.ref_token_mask = ref_token_mask.to(self.device)
            instance.input_referent_mask = input_referent_mask.to(self.device)
            batch.append(instance)
        max_lang_size = getattr(self.config, 'max_lang_size', 200 if self.config.vicuna_version ==
                                "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" or self.config.vicuna_version == "Tiny-Vicuna-1B" else 100)
        min_lang_size = min(getattr(self.config, 'min_lang_size', 100 if self.config.vicuna_version ==
                            "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" or self.config.vicuna_version == "Tiny-Vicuna-1B" else 50), max_lang_size)
        if self.subsample:
            num_samples = min(max(min(
                int(len(batch) * self.sample_rate), max_lang_size), min_lang_size), len(batch))
            batch = random.sample(batch, num_samples)
        if batch:
            inputs_embeds, attention_mask, labels = self._merge_input_ids_with_instance_features(
                batch=batch) 
        else:
            print("warning: no valid batch content, llm will not be updated, use dummy input")
            inputs_embeds = torch.randn(1, 256, 2048, requires_grad=True, device='cuda')

        gc.collect()
        torch.cuda.empty_cache()
        if num_of_truncated > 0:
            print(f"{num_of_truncated} of output is truncated")
        try:
            with torch.autocast("cuda"):
                if len(batch):
                    print(f"llm input embeds shape: {inputs_embeds.shape}") 
                    output = super().forward(
                        attention_mask=attention_mask, 
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    print("No batch detected! pass this iteration!")
                    return None
        except Exception as e:
            print("failed to forward")
            print(inputs_embeds.shape)
            print(e)
            print("Detected Error! pass this iteration!")
            return None

        output_last_hidden_states = output.hidden_states[-1].bfloat16()
        model_output = output 
        assert output_last_hidden_states.shape[:2] == labels.shape


        for instance, output_last_hidden_state in zip(batch, output_last_hidden_states):
            instance: grounded_3d_llm_data 
            s, e = instance.output_range
            instance.grouped_indices = instance.group_true_indices(output_last_hidden_state=output_last_hidden_state[s:e],
                                                                   MLP=self.model.hidden_state2query,
                                                                   ref_token_id=self.llama_tokenizer.ref_token_id) 

        loss_data_type = get_loss_for_each_type(output, batch) 
        output = model_output.logits
        lm_loss = model_output.loss 
        match_loss = self.calculate_grounding_loss(
            batch, use_single_ref_token=self.use_single_ref_token) 



        if self.config.use_reason_loss: 
            reason_loss = self.calculate_reasoning_loss(
                batch, output_last_hidden_states, attention_mask) 
            if torch.isnan(lm_loss + match_loss + reason_loss):
                print('Nan loss!')
                lm_loss = match_loss = reason_loss = model_output.logits.sum() * 0.
            return {
                "lm_loss": lm_loss,
                "match_loss": match_loss, 
                "reason_loss": reason_loss, 
                "model_output": model_output,
                **loss_data_type
            }
        
        else: 
            if torch.isnan(lm_loss + match_loss):
                print('Nan loss!')
                lm_loss = match_loss = model_output.logits.sum() * 0.
            return {
                "lm_loss": lm_loss,
                "match_loss": match_loss, 
                "model_output": model_output,
                **loss_data_type
            }

    def evaluate(
        self,
        input_text_list: list,
        batch_instance_queries_hidden_state: list,
        batch_instance_queries_normalized_embed: list,
        batch_eval_types: list,
        batch_thinking=None,
        max_new_tokens=150,
        use_mini_batch=False, 
        mini_batch_size=10,
        batch_gt_inst_ids: list = None,
        batch_out_text=None,
        output_logits=False,
        top_p=1.,
        repetition_penalty=1.2,
        length_penalty=1,
        text_only_output=False,  
        use_pre_reason_embed_for_inference=True,

    ):
        batch = []
        mini_batch = []

        # placeholder for GT
        if batch_out_text is None:
            batch_out_text = ["NONE"]*len(input_text_list)
        if batch_gt_inst_ids is None:
            batch_gt_inst_ids = [None]*len(input_text_list) 

        assert len(batch_out_text) == len(input_text_list)
        for input_text, instance_queries_hidden_state, instance_queries_normalized_embed, gt, eval_type, gt_instance_ids, gt_thinking in zip(input_text_list,
                                                                                                                                batch_instance_queries_hidden_state,
                                                                                                                                batch_instance_queries_normalized_embed,
                                                                                                                                batch_out_text,
                                                                                                                                batch_eval_types,
                                                                                                                                batch_gt_inst_ids,
                                                                                                                                batch_thinking
                                                                                                                                ):

            gt_instance_predicted_iou = None
            question_grounding_token_pos = None
            question_gt_query_ids = None

            if 'chat' not in eval_type: 
                if "scan2cap" in eval_type:
                    gt_instance_predicted_iou = gt_instance_ids[2]
                gt_instance_ids = gt_instance_ids[1]
                question_grounding_token_pos, question_gt_query_ids = gt_instance_ids.positives_question, gt_instance_ids.query_ids_question

            instance = grounded_3d_llm_data(input_text=input_text,
                                            instance_embed=instance_queries_normalized_embed,
                                            instance_feature=instance_queries_hidden_state,
                                            question_gt_query_ids=question_gt_query_ids,
                                            question_grounding_token_pos=question_grounding_token_pos,
                                            eval_type=eval_type,
                                            gt_instance_predicted_iou=gt_instance_predicted_iou,
                                            output_think_texts=gt_thinking, 
                                            )

            input_ids, input_referent_mask = instance.build_input_from_segments(use_system_prompt=self.config.use_system_prompt,
                                                                               tokenizer=self.llama_tokenizer,
                                                                               input_text=instance.input_text,
                                                                               prompts=self.prompts,
                                                                               use_input_referent=instance.use_input_referent,
                                                                               inference=True,
                                                                               question_grounding_token_pos=instance.question_grounding_token_pos,
                                                                               use_single_ref_token=self.use_single_ref_token,
                                                                               think=self.config.think,
                                                                               think_text=gt_thinking,
                                                                               )

            instance.input_ids = input_ids.to(self.device)
            instance.input_referent_mask = input_referent_mask.to(self.device)
            instance.output_hidden_states = []
            instance.gt = gt
            instance.eval_type = eval_type

            if not use_mini_batch:
                inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=[instance],
                                                                                             inference=True)
                instance.inputs_embeds = inputs_embeds
                instance.attention_mask = attention_mask
                batch.append(instance)
            else:
                mini_batch.append(instance)
                if len(mini_batch) == mini_batch_size:
                    inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=mini_batch,
                                                                                                 inference=True)
                    mini_batch = MiniBatchData(batch=mini_batch)
                    mini_batch.inputs_embeds = inputs_embeds
                    mini_batch.attention_mask = attention_mask
                    batch.append(mini_batch)
                    mini_batch = []

        if mini_batch and use_mini_batch:
            inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=mini_batch,
                                                                                         inference=True)
            mini_batch = MiniBatchData(batch=mini_batch)
            mini_batch.inputs_embeds = inputs_embeds
            mini_batch.attention_mask = attention_mask
            batch.append(mini_batch)

        # to avoid OOM error
        with torch.no_grad():
            for instance in batch: 
                gc.collect()
                torch.cuda.empty_cache()
                # try:
                common_params = {
                    'inputs_embeds': instance.inputs_embeds, 
                    'attention_mask': instance.attention_mask, 
                    'max_new_tokens': max_new_tokens,
                    'output_hidden_states': True,
                    'return_dict_in_generate': True,
                    'num_beams': self.beam_size,
                    'output_scores': True,
                    'do_sample': False,
                    'min_length': 1,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'length_penalty': length_penalty
                }
                if self.config.vicuna_version in ["TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", "Tiny-Vicuna-1B"]:
                    common_params['pad_token_id'] = self.llama_tokenizer.eos_token_id 

                with torch.autocast("cuda"):
                    outputs = self.generate(**common_params) 
                    output_ids = outputs.sequences 
                if use_pre_reason_embed_for_inference:
                    for index, id in enumerate(output_ids[-1]):
                        if id == self.llama_tokenizer.plan_token_id:
                            stage_1_output_ids = output_ids[-1][:index] 
                            stage_1_output_text = self.llama_tokenizer.decode(stage_1_output_ids.squeeze().tolist())
                            input_embeddings = self.model.get_input_embeddings() 
                            stage_1_output_embeds = input_embeddings(stage_1_output_ids)
                            question= instance.batch[0].input_text
                            plan_string = call_task_planner(stage_1_output_text, question)
                            plan_ids = self.llama_tokenizer.encode(plan_string, return_tensors="pt").to(stage_1_output_embeds.device)[:,1:]
                            plan_embeds = input_embeddings(plan_ids) 
                            stage_2_input_embeds = torch.cat((instance.inputs_embeds, stage_1_output_embeds.unsqueeze(0), plan_embeds), dim=1)
                            stage_2_attention_mask = torch.ones(
                                        (1, stage_2_input_embeds.shape[1]), device=self.device, dtype=bool) 

                            common_params = {
                                'inputs_embeds': stage_2_input_embeds,
                                'attention_mask': stage_2_attention_mask,
                                'max_new_tokens': max_new_tokens, 
                                'output_hidden_states': True,
                                'return_dict_in_generate': True,
                                'num_beams': self.beam_size,
                                'output_scores': True,
                                'do_sample': False,
                                'min_length': 1,
                                'top_p': top_p,
                                'repetition_penalty': repetition_penalty,
                                'length_penalty': length_penalty
                            }
                            if self.config.vicuna_version in ["TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", "Tiny-Vicuna-1B"]:
                                common_params['pad_token_id'] = self.llama_tokenizer.eos_token_id 
                            gc.collect()
                            torch.cuda.empty_cache()
                            with torch.autocast("cuda"):
                                outputs = self.generate(**common_params) 
                                stage_2_output_ids = outputs.sequences
                            output_ids = stage_2_output_ids 
                            full_output_ids = torch.cat((stage_1_output_ids.unsqueeze(0),plan_ids, stage_2_output_ids[:,1:]), dim=1) 
                            break 
                    show_output_text = self.llama_tokenizer.decode(full_output_ids.squeeze().tolist())
                
                last_hidden_states = extract_decoder_hidden_states(outputs)
                if use_mini_batch:
                    assert len(instance.batch) == output_ids.shape[0]
                    for _instance, ids, lash_hidden_state in zip(instance.batch, output_ids, last_hidden_states):
                        _instance.output_ids = ids.tolist()
                        _instance.output_hidden_states = self.model.hidden_state2query(
                            lash_hidden_state.squeeze(dim=0).bfloat16()).cpu()
                        _instance.output_text = self.llama_tokenizer.decode(
                            ids)
                else:
                    instance.output_ids = output_ids.squeeze().tolist()
                    instance.output_hidden_states = self.model.hidden_state2query(
                        last_hidden_states.squeeze(dim=0).bfloat16()).cpu()
                    instance.output_text = self.llama_tokenizer.decode(
                        instance.output_ids)
        torch.cuda.empty_cache
        
        if use_mini_batch:
            flattened_batch = []
            for instance in batch:
                flattened_batch += instance.batch
            batch = flattened_batch 

        template = {
            "grounding_start": None,
            "ref_token_pos": [],
            "ref_token_feature": [],
            "grounding_end": None,
            "closed": True,
            "match_result": None,
        }

        if output_logits:
            output_logits_list = []

        for bid, instance in enumerate(batch):
            intervals = []
            open_interval = False
            assistant_ids_len = self.prompts["rules"]["assistant"]["ids"].shape[0]
            current_interval = None
            assert len(
                instance.output_ids) == instance.output_hidden_states.shape[0]

            for idx, id in enumerate(instance.output_ids):
                if int(id) == self.gs_token_id: 
                    open_interval = True
                    current_interval = copy.deepcopy(template)
                    current_interval["grounding_start"] = idx-assistant_ids_len
                elif open_interval and id == self.ref_token_id:
                    current_interval["ref_token_feature"].append(
                        instance.output_hidden_states[idx])
                    current_interval["ref_token_pos"].append(
                        idx-assistant_ids_len)
                elif open_interval and int(id) == self.ge_token_id:
                    current_interval["grounding_end"] = idx-assistant_ids_len
                    current_interval["closed"] = True
                    open_interval = False
                    if current_interval["ref_token_feature"]: 
                        features_norm = F.normalize(torch.stack(
                            current_interval["ref_token_feature"]), p=2, dim=1)
                        embeddings_norm = F.normalize(
                            instance.instance_embed.cpu(), p=2, dim=1)
                        cosine_sim = torch.matmul(
                            features_norm.float(), embeddings_norm.T.float())
                        probs = cosine_sim
                        if not self.use_single_ref_token:
                            row_ind, col_ind = linear_sum_assignment(
                                probs.cpu().detach().numpy(), maximize=True)
                            best_match = col_ind.tolist()
                            best_match_probs = probs[row_ind, col_ind].tolist()
                        else: 
                            probs = torch.sigmoid(probs/self.t)
                            if output_logits and "detection" in instance.eval_type:
                                output_logits_list.append((bid, probs))
                            mask = probs > self.config.prediction_threshold 
                            if mask.any():
                                best_match = torch.nonzero(
                                    mask).squeeze(dim=1).tolist()
                                best_match = [i[1] for i in best_match]
                                best_match_probs = probs[mask].tolist()
                            else:
                                max_prob, max_index = torch.max(probs, dim=1) 
                                best_match = max_index.tolist() 
                                best_match_probs = max_prob.tolist()
                            k = min(10, probs.shape[1])
                            topk_values, topk_indices = torch.topk(probs, k=k, dim=1, largest=True, sorted=True)
                            top10_probs = topk_values.tolist()
                            top10_indices = topk_indices.tolist()
                        current_interval["match_result"] = best_match
                        current_interval["probs"] = best_match_probs
                        current_interval["topk_match_result"] = top10_probs
                        current_interval["topk_probs"] = top10_indices

                    else:
                        current_interval["match_result"] = None
                        current_interval["probs"] = None
                        current_interval["topk_match_result"] = None
                        current_interval["topk_probs"] = None
                    intervals.append(current_interval)
            if open_interval == True:
                current_interval["grounding_end"] = instance.output_hidden_states.shape[0]-1
                current_interval["closed"] = False
            instance.intervals = intervals
        out_json = []
        for instance in batch:
            item = {}
            item["output_language"] = instance.output_text
            item["input_language"] = instance.input_text
            item["grounding_result"] = []
            item["score"] = []
            item["topk_match_result"] = []
            item["topk_probs"] = []
            item["gt"] = instance.gt
            if 'scan2cap' in instance.eval_type or 'objdesc' in instance.eval_type:
                item['gt_predicted_iou'] = instance.gt_instance_predicted_iou
            if instance.intervals:
                for i in instance.intervals:
                    item["grounding_result"].append(i["match_result"])
                    item["score"].append(i["probs"]) 

                    item["topk_match_result"].append(i["topk_match_result"])
                    item["topk_probs"].append(i["topk_probs"]) 
            else:
                item["grounding_result"] = None
            out_json.append(item)
        if text_only_output:
            return out_json[0]["output_language"]
        if output_logits:
            return out_json, output_logits_list
        return out_json

