# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the original implementation of SetCriterion which will be deprecated in the next version.

We keep it here because our modified Criterion module is still under test.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized
import pytorch_metric_learning.losses 


def sigmoid_focal_loss(inputs, targets, num_boxes,t_weight, alpha: float = 0.25, gamma: float = 2, use_vlb = False):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    loss = loss.mean(1)
    if t_weight is not None and use_vlb:
        # print("using use_vlb")
        loss = loss * t_weight.unsqueeze(1)
    return loss.sum() / num_boxes


def vari_sigmoid_focal_loss(inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid().detach()  # pytorch version of RT-DETR has detach while paddle version not
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes



class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_vlb: bool = True,
        denoise_noobject_bbox: bool = False,
        sigmoid_focal_loss = False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        self.use_vlb = use_vlb
        self.denoise_noobject_bbox = denoise_noobject_bbox
        self.sigmoid_focal_loss = sigmoid_focal_loss
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
        ], "only support ce loss and focal loss for computing classification loss"

        if self.loss_class_type == "ce_loss":
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer("empty_weight", empty_weight)

        # self.contrastive_type = pytorch_metric_learning.losses.ContrastiveLoss()#exp30
        ###uncomment if using contrastive loss
        # self.contrastive_type = pytorch_metric_learning.losses.SoftTripleLoss(81,128)#exp31
        # self.contrastive_type = pytorch_metric_learning.losses.CosFaceLoss(81,128)#exp32
        # self.contrastive_type = pytorch_metric_learning.losses.TripletMarginLoss(triplets_per_anchor=100)#exp34
        # print("^^^^^^^^^^^^^^ contrastive_type", self.contrastive_type)
        print("^^^^^^^^^^^^^^ contrastive_version", 1)


    def loss_contrastive(self, outputs, targets, indices, num_boxes,t_weight,**kwargs):
        assert "pred_logits_emb" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits_emb"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        
        ##v1
        src_logits = src_logits.reshape(-1, 128) 
        target_classes = target_classes.reshape(-1) 
        contrastive_loss = self.contrastive_type(src_logits,target_classes)
        ##v2
        # contrastive_loss = 0
        # for i in range(src_logits.shape[0]):
        #     contrastive_loss+=self.contrastive_type(src_logits[i,:,:],target_classes[i,:])

        losses = {"loss_contrastive": contrastive_loss}
        return losses
        



    def loss_labels(self, outputs, targets, indices, num_boxes,t_weight,**kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score
        if self.use_alpha_scheduler and t_weight is not None:
            self.alpha = t_weight.view(-1,1,1)
        if self.use_vfl:
            loss_class = (
                vari_sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    target_score,
                    num_boxes=num_boxes,
                    t_weight = t_weight,
                    alpha=self.alpha,
                    gamma=self.gamma,
                ) * src_logits.shape[1]
            )
        else:
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    t_weight=t_weight,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    use_vlb = self.use_vlb,
                )
                * src_logits.shape[1]
            )
        losses = {"loss_class": loss_class}
        return losses
    
    def loss_labels_sigmoid_focal_loss(self, outputs, targets, indices, num_boxes,t_weight,**kwargs):
        # print("loss_labels_sigmoid_focal_loss")
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR
 
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
 
        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]
 
        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score
 
        loss_class = (
            vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score,
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, t_weight,**kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        # loss_bbox = F.mse_loss(src_boxes, target_boxes, reduction="none")
        if t_weight is not None and self.use_vlb:
            loss_bbox = loss_bbox * t_weight.unsqueeze(1).repeat(1, 4)
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        if self.denoise_noobject_bbox and t_weight is not None: ### todebug why t_weight is not None ? 
            class_weights = kwargs.get('class_weights',None)
            mask = torch.ones((outputs["pred_boxes"].shape[0],outputs["pred_boxes"].shape[1]), dtype=torch.bool)
            mask[idx] = False
            no_object_boxes = outputs["pred_boxes"][mask]
            no_object_boxes_target = torch.ones(no_object_boxes.shape, dtype=no_object_boxes.dtype).to(no_object_boxes.device)*0.5
            loss_bbox_no_object = F.l1_loss(no_object_boxes, no_object_boxes_target, reduction="none")
            if class_weights is not None and self.use_vlb:
                t_w = torch.ones((outputs["pred_boxes"].shape[0],outputs["pred_boxes"].shape[1]), dtype=loss_bbox_no_object.dtype).to(no_object_boxes.device) * class_weights.unsqueeze(1)
                t_w = t_w[mask]
                loss_bbox_no_object = loss_bbox_no_object * t_w.unsqueeze(1)
            losses["loss_bbox_noobject"] = loss_bbox_no_object.sum() / no_object_boxes.shape[0]


        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        if t_weight is not None and self.use_vlb:
            loss_giou = loss_giou * t_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses


    # def loss_diffusion(self, outputs, targets, indices, num_boxes, t_weight,**kwargs):
    #     """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
    #     targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
    #     The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    #     """
    #     idx = self._get_src_permutation_idx(indices)
    #     src_queries = outputs["query"][idx]
    #     target_queries = torch.cat([t["query"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
    #     loss_diffusion = F.l1_loss(src_queries, target_queries, reduction="none")
    #     loss_diffusion = loss_diffusion.mean(1)
    #     # loss_bbox = F.mse_loss(src_boxes, target_boxes, reduction="none")
    #     if t_weight is not None:
    #         loss_diffusion = loss_diffusion * t_weight
    #     losses = {}
    #     losses["loss_diffusion"] = loss_diffusion.sum() / num_boxes
    #     return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes,t_weight, **kwargs):
        if self.sigmoid_focal_loss:
            loss_map = {
                "class": self.loss_labels_sigmoid_focal_loss,
                "boxes": self.loss_boxes,
                # 'diffusion':self.loss_diffusion,
                "contrastive":self.loss_contrastive,
            }
        else:
            loss_map = {
                "class": self.loss_labels,
                "boxes": self.loss_boxes,
                # 'diffusion':self.loss_diffusion,
                "contrastive":self.loss_contrastive,
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, t_weight,**kwargs)

    def forward(self, outputs, targets, indices, t_weight, return_indices=False,**kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # indices = indices
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, t_weight, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                # indices = indices
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, t_weight, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
