import MinkowskiEngine as ME
import numpy as np
import torch
from random import random

from transformers import AutoTokenizer, BertConfig


class batch_lang_data:
    def __init__(self,
                 batch_concat_texts=[],
                 tokenized=None,
                 batch_num_concat_texts=None,
                 raw_phrases=None,
                 flatten_lang_token_inst_id_pairs=None,
                 raw_lang_types=None,
                 concat_lang_positive_maps=None,
                 unflatten_lang_token_inst_id_pairs=None):

        # extra_query_texts
        self.batch_concat_texts = batch_concat_texts
        self.tokenized = tokenized
        self.batch_num_concat_texts = batch_num_concat_texts
        self.raw_phrases = raw_phrases
        self.flatten_lang_token_inst_id_pairs = flatten_lang_token_inst_id_pairs
        self.raw_lang_types = raw_lang_types

        # extra_instance_masks
        self.concat_lang_positive_maps = concat_lang_positive_maps
        self.unflatten_lang_token_inst_id_pairs = unflatten_lang_token_inst_id_pairs

    def __len__(self):
        return len(self.batch_concat_texts)

    def print(self):
        print('extra_query_texts:')
        print(f'- batch_concat_texts: [{len(self.batch_concat_texts)}, str]')
        print(f'- tokenized: {list(self.tokenized.input_ids.shape)}')
        print(f'- batch_num_concat_texts = {self.batch_num_concat_texts}')
        print(
            f'- raw_phrases: [{len(self.raw_phrases)}, str] = shape of {[(len(i), str) for i in self.raw_phrases]}')
        print(
            f'- flatten_lang_token_inst_id_pairs: [{len(self.flatten_lang_token_inst_id_pairs)}, ?] = shape of {[(len(i), "?", 2) for i in self.flatten_lang_token_inst_id_pairs]}')
        print(
            f'- raw_lang_types: [{len(self.raw_lang_types)}, ?] = shape of {[len(i) for i in self.raw_lang_types]}')
        print('extra_instance_masks:')
        print(
            f'- concat_lang_positive_maps: [{len(self.concat_lang_positive_maps)}, ?] = shape of {[list(i.shape) for i in self.concat_lang_positive_maps]}')
        print(
            f'- unflatten_lang_token_inst_id_pairs: [{len(self.unflatten_lang_token_inst_id_pairs)}, ?] = shape of {[i.shape for i in self.unflatten_lang_token_inst_id_pairs]}')


class VoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
        sample_class_labels=False,
        bert_path="./bert-base-uncased"
    ):
        assert task in [
            "instance_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries
        self.sample_class_labels = sample_class_labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_path, model_max_length=256)

    def __call__(self, batch):
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            num_queries=self.num_queries,
            tokenizer=self.tokenizer,
            sample_class_labels=self.sample_class_labels,
        )


def voxelize(
    batch,
    ignore_label,
    voxel_size,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
    tokenizer,
    sample_class_labels=False,
):
    (
        coordinates,
        features,
        labels,
        original_labels,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
    ) = ([], [], [], [], [], [], [], [], [])
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }
    extra_lang = []
    extra_qa = []

    full_res_coords = []

    for sample in batch:
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[4])
        original_normals.append(sample[5])

        if len(sample) > 8:
            extra_lang.append(sample[8])
            extra_qa.append(sample[9])

        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(
                input_dict["labels"][i][:, 0],
                return_index=True,
                return_inverse=True,
            )
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                if input_dict["labels"][i].shape[1] > 2:  # no segment
                    # TODO BIGGER CHANGE CHECK!!!
                    _, ret_index, ret_inv = np.unique(
                        input_dict["labels"][i][:, 2],
                        return_index=True,
                        return_inverse=True,
                    )
                    input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                    input_dict["segment2label"].append(
                        input_dict["labels"][i][ret_index][:, :-1]
                    )

    if "labels" in input_dict:
        list_labels = input_dict["labels"]
        # for list_label,original_label in zip(list_labels,original_labels):
        #     instance_label = list_label[:,1]
        #     instance_ids = instance_label.unique()
        #     origin_ids = torch.from_numpy(np.unique(original_label[:,1])).long() 
        #     assert torch.equal(instance_ids, origin_ids), f"instance_ids: {instance_ids}!=origin_ids: {origin_ids}"
        target = []
        target_full = []

        assert len(list_labels[0].shape) != 1
        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id]
                        == label_ids.unsqueeze(1),
                    }
                )
        else:
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append(
                        {"point2segment": input_dict["labels"][i][:, 0]}
                    )
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long()
                        }
                    )
            else:
                target = get_instance_masks(
                    list_labels,
                    list_segments=input_dict["segment2label"],
                    task=task,
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                    extra_lang=extra_lang,
                    extra_qa=extra_qa,
                )

                batch_num_concat_texts = [
                    len(lang_bid.concat_texts) for lang_bid in extra_lang]
                batch_concat_texts = [
                    i for lang_bid in extra_lang for i in lang_bid.concat_texts]

                concat_lang_positive_maps = [None] * len(extra_lang)
                raw_phrases = [[]] * len(extra_lang)
                unflatten_lang_token_inst_id_pairs = [None] * len(extra_lang)
                flatten_lang_token_inst_id_pairs = [None] * len(extra_lang)
                raw_lang_types = [None] * len(extra_lang)

                if len(batch_concat_texts) != 0:
                    tokenized = tokenizer(
                        batch_concat_texts, padding='longest', truncation=True, return_tensors='pt')
                    num_max_tokens = np.asarray(tokenized.input_ids).shape[1]

                    total_concat_texts = 0
                    for bid in range(len(extra_lang)):
                        each_lang_positive_map = []
                        lang_token_inst_id_pairs = []
                        raw_lang_token_inst_id_pairs = []
                        raw_lang_type = []

                        for lang_id, (gt_inst_ids, positives) in enumerate(zip(
                                extra_lang[bid].concat_gt_insts, extra_lang[bid].concat_positives)):
                            assert len(gt_inst_ids) == len(positives)

                            raw_lang_token_inst_id_pairs.append([])
                            raw_lang_type.append(
                                extra_lang[bid].concat_types[lang_id])

                            positives = np.asarray(positives).astype(int)
                            if len(positives) == 0:
                                continue
                            # assume concat_text_id are same
                            assert np.all(positives[0, 0] == positives[:, 0])

                            for inst_ids_per_phrase, (concat_text_id, beg, end) in zip(gt_inst_ids, positives):
                                try:
                                    beg_token = tokenized.char_to_token(
                                        total_concat_texts + concat_text_id, beg)
                                    end_token = tokenized.char_to_token(
                                        total_concat_texts + concat_text_id, end)
                                    if beg_token is None:
                                        try:
                                            beg_token = tokenized.char_to_token(
                                                total_concat_texts + concat_text_id, beg + 1)
                                            if beg_token is None:
                                                beg_token = tokenized.char_to_token(
                                                    total_concat_texts + concat_text_id, beg + 2)
                                        except:
                                            beg_token = None
                                    if end_token is None:
                                        try:
                                            end_token = tokenized.char_to_token(
                                                total_concat_texts + concat_text_id, end + 1)
                                            if end_token is None:
                                                end_token = tokenized.char_to_token(
                                                    total_concat_texts + concat_text_id, end + 2)
                                        except:
                                            end_token = None
                                    if beg_token is None or end_token is None:  # If no beg/end token is found, then ignore this one
                                        continue
                                except:
                                    continue

                                lang_positive_map_i = torch.zeros(
                                    (batch_num_concat_texts[bid], num_max_tokens), dtype=bool)  # all texts
                                lang_positive_map_i[concat_text_id,
                                                    beg_token:end_token] = 1

                                each_lang_positive_map.append(
                                    lang_positive_map_i)
                                raw_phrases[bid].append(
                                    batch_concat_texts[total_concat_texts + concat_text_id][beg:end])

                                pairs = []
                                for inst_id in inst_ids_per_phrase:
                                    if inst_id is None:
                                        continue
                                    pairs.append(
                                        (len(each_lang_positive_map)-1, inst_id))
                                lang_token_inst_id_pairs.extend(pairs)
                                raw_lang_token_inst_id_pairs[-1].extend(pairs)

                        if len(each_lang_positive_map) > 0:
                            concat_lang_positive_maps[bid] = torch.stack(
                                each_lang_positive_map, dim=-1)
                        else:
                            concat_lang_positive_maps[bid] = torch.zeros(
                                [batch_num_concat_texts[bid], num_max_tokens, 0], dtype=bool)
                        unflatten_lang_token_inst_id_pairs[bid] = np.asarray(
                            lang_token_inst_id_pairs)
                        flatten_lang_token_inst_id_pairs[bid] = raw_lang_token_inst_id_pairs
                        raw_lang_types[bid] = raw_lang_type

                        total_concat_texts += batch_num_concat_texts[bid]

                    assert np.all(batch_num_concat_texts[0] == np.asarray(
                        batch_num_concat_texts[0:]))
                    extra_lang = batch_lang_data(
                        batch_concat_texts, tokenized, batch_num_concat_texts, raw_phrases, flatten_lang_token_inst_id_pairs, raw_lang_types,
                        concat_lang_positive_maps, unflatten_lang_token_inst_id_pairs,
                    )
                else:
                    extra_lang = batch_lang_data()

                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:,
                                                                         2] if input_dict["labels"][i].shape[1] > 2 else None
                if "train" not in mode:
                    target_full = get_instance_masks(
                        [torch.from_numpy(l) for l in original_labels],
                        task=task,
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                    )
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(
                            original_labels[i][:, 2]
                        ).long() if original_labels[i].shape[1] > 2 else None
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    if "train" not in mode:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                target_full,
                original_colors,
                original_normals,
                original_coordinates,
                idx,
                extra_lang=extra_lang,
                extra_qa=extra_qa,
            ),
            target,
            [sample[3] for sample in batch],
        )
    else:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                extra_lang=extra_lang,
                extra_qa=extra_qa,
            ),
            target,
            [sample[3] for sample in batch],
        )


def get_instance_masks(
    list_labels,
    task,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
    extra_lang=None,
    extra_qa=None
):
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        num_valid_instance = 0
        instance_mapping = {} # 这是个字典
        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            tmp = list_labels[batch_id][
                list_labels[batch_id][:, 1] == instance_id
            ]
            label_id = tmp[0, 0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            label_ids.append(label_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            if list_segments:
                segment_mask = torch.zeros(
                    list_segments[batch_id].shape[0]
                ).bool()
                segment_mask[
                    list_labels[batch_id][
                        list_labels[batch_id][:, 1] == instance_id
                    ][:, 2].unique()
                ] = True
                segment_masks.append(segment_mask)

            instance_mapping[int(instance_id)] = num_valid_instance # 构建对应的关系
            num_valid_instance += 1 
        try:
            # change extra_lang labels
            if extra_lang is not None and len(extra_lang) > 0 and hasattr(extra_lang[batch_id], 'concat_texts') and len(extra_lang[batch_id].concat_texts) > 0:
                extra_lang[batch_id].remap_inst_ids(instance_mapping) # 根据这个mapping来remap

            if extra_qa is not None and len(extra_qa) > 0:
                for lang_info in extra_qa[batch_id]:
                    lang_info.remap_inst_ids(instance_mapping)
        except Exception as e:
            print("Error in remapping instance ids")
            from IPython import embed
            embed()
            pass
        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        masks = torch.stack(masks)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        assert task == 'instance_segmentation'
        # clamp to max(1-2, 0) = 0 (chair), max(3-2,0)=1 (table)
        l = torch.clamp(label_ids - label_offset, min=0)

        if list_segments:
            target.append(
                {
                    "labels": l,
                    "masks": masks,
                    "segment_mask": segment_masks,
                }
            )
        else:
            target.append({"labels": l, "masks": masks})

    return target


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        full_res_coords=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
        extra_lang=None,
        extra_qa=None
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
        self.extra_lang = extra_lang
        self.extra_qa = extra_qa


def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip(
                    'axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    assert np.all(np.fabs(axis_align_matrix[3, :3]) < 1e-8)
    return axis_align_matrix


def concatenate_texts_with_separator(tokenizer, raw_texts, max_batch_tokens, num_concat_texts, max_tokens,
                                     raw_texts_poschars, raw_texts_posinsts, raw_texts_type=None, shuffle=False, text_separator='. ', concat=True):
    assert tokenizer.model_max_length >= max_batch_tokens
    # assert max([len(i) for i in tokenizer(raw_texts).input_ids]) <= max_tokens

    if shuffle:
        random_text_indices = np.arange(len(raw_texts))
        np.random.shuffle(random_text_indices)
        raw_texts_remap = np.zeros((len(raw_texts)), dtype=int)
        raw_texts_remap[random_text_indices] = np.arange(
            len(random_text_indices))
    else:
        random_text_indices = list(range(len(raw_texts)))
        raw_texts_remap = np.arange(len(random_text_indices))

    if len(raw_texts) > 0:
        raw_texts_input_ids_length = np.asarray(
            [len(i) for i in tokenizer(raw_texts).input_ids])

    concat_texts = []
    concat_texts_pos_tokens = []
    concat_texts_pos_insts = []
    concat_texts_type = []
    text = ''
    char_id = 0
    len_concat_compute = 2
    for i, label_idx in enumerate(random_text_indices):
        if len(concat_texts) < num_concat_texts or not concat:
            text += raw_texts[label_idx]
            len_concat_compute += raw_texts_input_ids_length[label_idx] - 2
            concat_texts_pos_tokens.append([])
            concat_texts_pos_insts.append([])
            if raw_texts_type is not None:
                concat_texts_type.append(raw_texts_type[label_idx])
            if raw_texts_poschars is not None and len(raw_texts_poschars[label_idx]) > 0:
                for j, ((beg, end), inst_ids) in enumerate(zip(raw_texts_poschars[label_idx], raw_texts_posinsts[label_idx])):
                    concat_texts_pos_tokens[-1].append(
                        (len(concat_texts), char_id + beg, char_id + end))
                    concat_texts_pos_insts[-1].append(inst_ids)

            text += text_separator
            char_id = len(text)

            if not concat:
                concat_texts.append(text)
                text = ''
                char_id = 0
            else:
                if i == len(random_text_indices) - 1:
                    concat_texts.append(text)
                    text = ''
                    char_id = 0
                    len_concat_compute = 2
                else:
                    if len_concat_compute + raw_texts_input_ids_length[random_text_indices[i + 1]] - 2 >= max_batch_tokens:
                        # avoid too long tokenizer
                        if i < len(random_text_indices) - 1 and raw_texts_input_ids_length[random_text_indices[i + 1]] >= max_batch_tokens:
                            while len(tokenizer(raw_texts[random_text_indices[i + 1]] + text_separator).input_ids) >= max_batch_tokens:
                                num_drop = max_batch_tokens - \
                                    len(tokenizer(
                                        raw_texts[random_text_indices[i + 1]] + text_separator).input_ids) + 1
                                raw_texts[random_text_indices[i + 1]] = ' '.join(
                                    raw_texts[random_text_indices[i + 1]].split(' ')[:-(num_drop + 1)] + ['.'])

                        concat_texts.append(text)
                        text = ''
                        char_id = 0
                        len_concat_compute = 2

    if len(concat_texts) < num_concat_texts and concat:
        concat_texts += [''] * (num_concat_texts - len(concat_texts))
    concat_texts_pos_tokens = np.asarray(concat_texts_pos_tokens, dtype=object)[
        raw_texts_remap[:len(concat_texts_pos_tokens)]]
    concat_texts_pos_insts = np.asarray(concat_texts_pos_insts, dtype=object)[
        raw_texts_remap[:len(concat_texts_pos_insts)]]
    if raw_texts_type is not None:
        concat_texts_type = np.asarray(concat_texts_type, dtype=object)[
            raw_texts_remap[:len(concat_texts_type)]]

    return concat_texts, concat_texts_pos_tokens, concat_texts_pos_insts, concat_texts_type
