# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

def binary_focal_loss_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            (num_queries, num_class).
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

    Returns:
        torch.Tensor: cls_cost value with weight
    """
    cls_pred = cls_pred.flatten(1)
    gt_labels = gt_labels.flatten(1).float()
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (
        1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (
        1 - cls_pred).pow(gamma)

    cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
        torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
    return cls_cost

def smooth_l1(diff, beta = 1.0):
    loss = 0
    mask = (diff.abs() < beta)
    loss += mask * (0.5*diff**2 / beta)
    loss += (~mask) * (diff.abs() - 0.5*beta)
    return loss.mean(dim=-1)


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        num_queries: int = 0, 
        language_mode: bool = False,
        softmax_mode: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.num_queries = num_queries

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points
        self.language_mode = language_mode
        self.softmax_mode = softmax_mode

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        bs = len(outputs["pred_logits"])

        indices = []
        
        if self.language_mode:
            batch_concat_positive_map, batch_token_inst_ids_pair = outputs['extra_lang'].concat_lang_positive_maps, outputs['extra_lang'].unflatten_lang_token_inst_id_pairs

        # Iterate through batch size
        for b in range(bs):
            pred_logits_bid = outputs["pred_logits"][b]
            pred_masks_bid = outputs['pred_masks'][b]

            if self.language_mode and not self.softmax_mode:
                out_mask = pred_masks_bid.T  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                tgt_mask = targets[b][mask_type].to(out_mask)

                if self.num_points != -1:
                    point_idx = torch.randperm(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )[: int(self.num_points * tgt_mask.shape[1])]
                    # point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
                else:
                    # sample all points
                    point_idx = torch.arange(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    )

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    )

                num_targets = tgt_mask.shape[0]
                num_positive_tokens = pred_logits_bid.shape[1]
                token_inst_ids_pair = batch_token_inst_ids_pair[b] # a list of (pos_token_id, inst_id)
                logit_labels = pred_logits_bid.new_zeros((num_targets, num_positive_tokens))
                if len(token_inst_ids_pair) > 0:
                    logit_labels[token_inst_ids_pair[:, 1], token_inst_ids_pair[:,0]] = 1
                else:
                    print('Only negative examples')

                positive_map = batch_concat_positive_map[b].flatten(0, 1)
                invalid_text_token = positive_map.sum(dim=0) < 1e-8

                cost_class = binary_focal_loss_cost(
                    pred_logits_bid[:, ~invalid_text_token],
                    logit_labels[:, ~invalid_text_token],
                )
                
                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )

                C = C.reshape(self.num_queries, -1).cpu()

                indices.append(linear_sum_assignment(C))

            elif not self.language_mode and not self.softmax_mode: 
                out_mask = pred_masks_bid.T  
                tgt_mask = targets[b][mask_type].to(out_mask) 

                if self.num_points != -1:
                    point_idx = torch.randperm(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )[: int(self.num_points * tgt_mask.shape[1])]

                else:
                    point_idx = torch.arange(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    cost_mask = batch_sigmoid_ce_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    ) 

                    cost_dice = batch_dice_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    ) 

                num_targets = tgt_mask.shape[0]
                num_positive_tokens = pred_logits_bid.shape[1] 
                tgt_ids = targets[b]["labels"].clone()
                filter_ignore = tgt_ids == 253
                tgt_ids[filter_ignore] = 0

                logit_labels = pred_logits_bid.new_zeros((num_targets, num_positive_tokens))
                logit_labels[torch.arange(num_targets), tgt_ids] = 1

                cost_class = binary_focal_loss_cost(
                    pred_logits_bid,
                    logit_labels,
                ) 
                cost_class[:, filter_ignore] = 0.
                
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                ) 

                C = C.reshape(self.num_queries, -1).cpu()

                indices.append(linear_sum_assignment(C)) 

            else:
                out_prob = pred_logits_bid.softmax(
                    -1
                ) 
                tgt_ids = targets[b]["labels"].clone()
                filter_ignore = tgt_ids == 253
                tgt_ids[filter_ignore] = 0
                cost_class = -out_prob[:, tgt_ids]
                cost_class[
                    :, filter_ignore
                ] = (
                    -1.0
                ) 

                out_mask = pred_masks_bid.T 
                tgt_mask = targets[b][mask_type].to(out_mask)

                if self.num_points != -1:
                    point_idx = torch.randperm(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )[: int(self.num_points * tgt_mask.shape[1])]

                else:

                    point_idx = torch.arange(
                        tgt_mask.shape[1], device=tgt_mask.device
                    )

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()

                    cost_mask = batch_sigmoid_ce_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    )


                    cost_dice = batch_dice_loss_jit(
                        out_mask[:, point_idx], tgt_mask[:, point_idx]
                    )

                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )

                C = C.reshape(self.num_queries, -1).cpu()

                indices.append(linear_sum_assignment(C))
        
        return [
            [
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            ]
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
