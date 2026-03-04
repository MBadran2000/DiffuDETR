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

import torch

## Diffu_dino
# from detrex.modeling.criterion import SetCriterion
from .diffu_criterion import SetCriterion

from detrex.utils import get_world_size, is_dist_avail_and_initialized


class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,

        use_vlb: bool= True,
        use_matcher: bool = False,
        use_encoder_loss: bool = False,
        use_count_loss = False,
        denoise_noobject_bbox: bool = False,
        sigmoid_focal_loss = False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls


        self.use_vlb = use_vlb
        self.denoise_noobject_bbox = denoise_noobject_bbox
        self.use_matcher = use_matcher
        self.use_encoder_loss = use_encoder_loss
        self.use_count_loss = use_count_loss
        self.sigmoid_focal_loss = sigmoid_focal_loss
        print("sigmoid_focal_loss ",sigmoid_focal_loss)


    def forward(self, outputs, targets, indices=None, t_weight=None, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)

        ## count_loss
        # if self.use_count_loss:
        #     topk_values, topk_indexes = torch.topk(outputs['pred_logits'], 1, dim=-1) # [b, q, 1] , [b, q, 1]) 
        #     topk_indexes[topk_values.squeeze(-1)>-1]= 80 ###### debug ????
        #     topk_indexes = topk_indexes.squeeze(-1)
        #     pred_class_no = []
        #     gt_class_no = []
        #     for i in range(len(targets)):
        #         cur_img_pred = []
        #         cur_img_gt = []
        #         for j in range(0,80):
        #             cur_img_pred.append((topk_indexes[i] == j ).sum())
        #             cur_img_gt.append((targets[i]['labels']== j ).sum())
        #         pred_class_no.append(cur_img_pred)
        #         gt_class_no.append(cur_img_gt)
        #     pred_class_no = torch.tensor(pred_class_no)
        #     gt_class_no = torch.tensor(gt_class_no)

        #     count_loss = {}
        #     count_loss['loss_count'] =  nn.L1Loss()(pred_class_no.float(), gt_class_no.float())*20 ## L1Loss MSELoss HuberLoss



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
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,t_weight, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, t_weight , **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if self.use_encoder_loss:
            if "enc_outputs" in outputs:
                enc_outputs = outputs["enc_outputs"]
                if self.two_stage_binary_cls:
                    for bt in targets:
                        bt["labels"] = torch.zeros_like(bt["labels"])
                indices = self.matcher(enc_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, None, **kwargs)
                    l_dict = {k + "_enc": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
