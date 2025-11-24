# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchvision.ops import sigmoid_focal_loss
import torch.distributed as dist

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
        language_mode=False,
        softmax_mode=False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.softmax_mode = softmax_mode
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.language_mode = language_mode

    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs

        if self.language_mode and not self.softmax_mode:
            loss_ce = 0.
            batch_concat_positive_map, batch_token_inst_ids_pair = outputs['extra_lang'].concat_lang_positive_maps, outputs['extra_lang'].unflatten_lang_token_inst_id_pairs
            for bid, (pred_logit, indice) in enumerate(zip(outputs['pred_logits'], indices)):
                token_inst_ids_pair = batch_token_inst_ids_pair[bid]
                
                map_target_to_query = torch.zeros((targets[bid]['labels'].shape[0]), dtype=torch.long) - 1
                map_target_to_query[indice[1]] = indice[0]
                # valid_map = map_target_to_query > -1
                # map_target_to_query[~valid_map] = 0

                logits_target = torch.zeros_like(pred_logit) # [num_query, num_pos_token_i]
                if len(token_inst_ids_pair) > 0:
                    logits_target[map_target_to_query[token_inst_ids_pair[:, 1]], token_inst_ids_pair[:, 0]] = 1
                # print('loss class targets unique id', np.unique(token_inst_ids_pair[:, 0]))

                positive_map = batch_concat_positive_map[bid].flatten(0, 1)
                invalid_text_token = positive_map.sum(dim=0) < 1e-8

                loss_ce += sigmoid_focal_loss(pred_logit[:, ~invalid_text_token], logits_target[:, ~invalid_text_token], reduction='sum') / (logits_target[:, ~invalid_text_token].sum() + 1.)
            loss_ce /= len(outputs['pred_logits'])
        elif not self.language_mode and not self.softmax_mode:
            if isinstance(outputs["pred_logits"], list):
                src_logits = torch.stack(outputs["pred_logits"], 0).float()
            else:
                src_logits = outputs["pred_logits"].float()

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)]
            )
            target = torch.zeros_like(src_logits)
            weight = torch.ones_like(src_logits)
            ignore_class = target_classes_o == 253
            target_classes_o[ignore_class] = 0
            target[idx[0], idx[1], target_classes_o] = 1
            weight[idx[0][ignore_class], idx[1][ignore_class], :] = 0
            invalid = src_logits < -1e8
            loss_ce = sigmoid_focal_loss(src_logits[~invalid]*weight[~invalid], target[~invalid]*weight[~invalid], reduction='sum') / (target.sum() + 1.)
        else:
            if isinstance(outputs["pred_logits"], list):
                src_logits = torch.stack(outputs["pred_logits"], 0).float()
            else:
                src_logits = outputs["pred_logits"].float()

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)]
            )
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o

            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight,
                ignore_index=253,
            )
        
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []
        
        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))

        # del target_mask
        ret = {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

        return ret 

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type) 
        self.indices = indices # saved for llm

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device='cuda',
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_masks, mask_type
                )
            )
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                        mask_type,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
