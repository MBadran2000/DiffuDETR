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

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb


from .bbox_embedd import BBoxEmbed , ClassEmbed , TimeStepBlock
from functools import partial
from .ldm.modules.diffusionmodules.util import timestep_embedding , linear
from .ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps

from detectron2.layers import batched_nms

## Diffu_dino
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    if len(t.shape) > 1:
        t = t.squeeze()
    out = a.gather(-1, t.squeeze())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
####

class DIFFUSION_DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        old_schedule= True,
        noisy_gt = False,
        v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        timesteps = 1000,
        sampling_steps = 25,
        beta_schedule = 'linear',
        given_betas=None,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        dino_type = ""
    ):
        super().__init__()
        assert dino_type == "diffu_det", "check init file ******************"

        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes +1 , embed_dim) ##### +1  for Diffu_Dino
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"


        ## Diffu_dino
        self.v_posterior = v_posterior
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_steps
        self.ddim_sampling_eta = 0.0
        
        self.parameterization = "x0"
        # self.register_schedule_old(timesteps)
        if old_schedule:
            print("*************** schedule old")
            self.register_schedule_old(timesteps)
        else:
            print("*************** schedule ldm")
            self.register_schedule_ldm(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # print("******************************* old")

        self.parameterization = "x0"
        # self.register_schedule_ldm(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
        #                        linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # self.make_schedule()
        self.noisy_gt = noisy_gt
        print("$$$$$$$$$$$$$$$ diffu det noise")
        self.scale = 2

        # self.drop_gt = drop_gt
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True



    def register_schedule_old(self,time_steps):
        betas = cosine_beta_schedule(time_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # snr = alphas_cumprod / (1 - alphas_cumprod)
        # maybe_clipped_snr = snr.clone().clamp(max = 2)
        # self.register_buffer('loss_weight', maybe_clipped_snr,persistent=False)

        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('loss_weight', lvlb_weights, persistent=False)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def register_schedule_ldm(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('loss_weight', lvlb_weights, persistent=False)
        assert not torch.isnan(self.loss_weight).all()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # def apply_box_noise(self,boxes: torch.Tensor,t):
    #     """
    #     Args:
    #         boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
    #             shape ``(num_boxes, 4)``
    #         box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    #     """
    #     # print(boxes)
    #     # t = t = torch.full((1,), t).long() 
    #     # print(boxes.shape)
    #     # diff = torch.zeros_like(boxes)
    #     # diff[:, :2] = boxes[:, :2] 
    #     # diff[:, 2:] = boxes[:, 2:]
    #     # boxes_noise = (torch.randn_like(boxes)+2)/4#torch.mul(((torch.randn_like(boxes)+1.5)/3), diff)
    #     boxes_noise = torch.randn_like(boxes)
    #     boxes = inverse_sigmoid(boxes)
    #     # print("boxes_noise",boxes_noise)
    #     # # boxes_noise =
    #     # print(diff)
    #     # print('diff' , diff)
    #     # print('boxes_noise' , boxes_noise)
    #     boxes = self.q_sample(x_start = boxes , t = t, noise = boxes_noise)
    #     boxes = boxes.sigmoid()
    #     # boxes = boxes.clamp(min=0.0, max=1.0)
    #     return boxes , boxes_noise.sigmoid()

    def apply_box_noise(self,boxes: torch.Tensor,t):
        """
        Args:
            boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
                shape ``(num_boxes, 4)``
            box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        """
        noise = torch.randn_like(boxes)
        
        gt_boxes = (boxes * 2. - 1.) * self.scale
        # x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=gt_boxes, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = x
        return diff_boxes, noise

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = images.image_sizes[img_id]
                img_masks[img_id, :(img_h-1), :(img_w-1)] = 0
        # else:
        #     batch_size, _, H, W = images.tensor.shape
        #     img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            ## Diffu_dino
            # targets = self.prepare_targets(gt_instances)
            targets, new_targets_diffusion= self.prepare_targets(gt_instances)

            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
            noisy_queries, init_query_points , queries, t , indices, num_gts = self.process_targets(targets, new_targets_diffusion)
        else:
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            # input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
            #     targets,
            #     dn_number=self.dn_number,
            #     label_noise_ratio=self.label_noise_ratio,
            #     box_noise_scale=self.box_noise_scale,
            #     num_queries=self.num_queries,
            #     num_classes=self.num_classes,
            #     hidden_dim=self.embed_dim,
            #     label_enc=self.label_enc,
            # )
            # input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
            return self.ddim_sample(batched_inputs, multi_level_feats, multi_level_masks, multi_level_position_embeddings,images)

        
        query_embeds = (input_query_label, input_query_bbox)
        query_embeds_diffusion = (noisy_queries, init_query_points)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            query_embeds_diffusion = query_embeds_diffusion,
            time_steps= t,
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        lvlb_class_weights = extract(self.loss_weight, t, [batch_size])
        t_weight = extract(self.loss_weight, t, [batch_size , self.num_queries])
        t_weight = [t_weight[i].expand(len(indices[i][0])) for i in range(batch_size)]
        t_weight = torch.cat(t_weight)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            
            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta, indices,t_weight, class_weights=lvlb_class_weights)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).to(self.device) for t in targets]
        # known = [(torch.ones_like(t["labels"])) for t in targets]

        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().to(self.device).unsqueeze(0).repeat(dn_number, 1)
        )
        # positive_idx = (
        #     torch.tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        # )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().to(self.device).unsqueeze(1)
        # positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)

        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).to(self.device) * box_noise_scale
            # known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to(self.device)
        # m = known_labels_expaned.long()

        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).to(self.device)
        padding_bbox = torch.zeros(pad_size, 4).to(self.device)
        # padding_label = torch.zeros(pad_size, hidden_dim)
        # padding_bbox = torch.zeros(pad_size, 4)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to(self.device)
        # map_known_indice = torch.tensor([])

        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(self.device) < 0
        # attn_mask = torch.ones(tgt_size, tgt_size) < 0

        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes, final = True):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        # print(box_cls)
        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        # boxes = box_pred

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            if self.use_ensemble and self.sampling_timesteps > 1 and not final:
                return result.pred_boxes.tensor, scores_per_image, labels_per_image
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image


            results.append(result)
        return results

    # def prepare_targets(self, targets):
    #     new_targets = []
    #     for targets_per_image in targets:
    #         h, w = targets_per_image.image_size
    #         image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
    #         gt_classes = targets_per_image.gt_classes
    #         gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
    #         gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
    #         new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
    #     return new_targets

    ## Diffu_dino
    def prepare_targets(self, targets):
        new_targets = []
        num_gts = []
        new_targets_diffusion = []
        for targets_per_image in targets:
            no_object_bbox , no_object_label = torch.tensor([0.5,0.5,0.5,0.5],device=self.device), torch.tensor([self.num_classes],device= self.device)
            # no_object_bbox , no_object_label = torch.tensor([1,1,1,1],device=self.device), torch.tensor([80],device= self.device)

            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            num_gt = len(gt_boxes)
            no_object_label = no_object_label.repeat(self.num_queries - num_gt, 1) # 295 , 256
            no_object_bbox = no_object_bbox.repeat(self.num_queries - num_gt , 1) # 295 , 4

            ####### noisy ground truth
            if self.noisy_gt:
                no_object_bbox = torch.randn(no_object_bbox.shape, device=self.device)
                no_object_bbox = no_object_bbox.sigmoid()
            ######

            gt_boxes_diffusion = torch.cat([gt_boxes , no_object_bbox])
            gt_classes_diffusion = torch.cat([gt_classes , no_object_label.squeeze()])
            # noised_gtboxes, noise , t  = self.prepare_for_diffusion(gt_boxes)
            new_targets_diffusion.append({"labels": gt_classes_diffusion, "boxes": gt_boxes_diffusion})
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            # num_gts.append(num_gt)
        return new_targets , new_targets_diffusion



    @torch.no_grad()
    def ddim_sample(self, batched_inputs, multi_level_feats,multi_level_masks,
                    multi_level_position_embeddings,images,
                    clip_denoised=True, do_postprocess=True):
        N = len(batched_inputs)
        # shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        t = torch.zeros((N,), device=self.device, dtype=torch.long)
        query = [None,  None]
        # prepare for loss computation
        (memory,
        mask_flatten ,
        spatial_shapes ,
        level_start_index,
        valid_ratios, 
        init_reference,
        target_unact) = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings,query, attn_masks=None ,time_steps=t,query_embeds_diffusion= [None,None], is_train =False)
        # query = torch.randn((N,self.num_queries,self.embed_dim), device=self.device)
        if self.transformer.learnt_init_query:
            if self.transformer.tgt_embed_new:
                query = self.transformer.tgt_embed.weight[None].repeat(N, self.num_queries, 1)
            else:
                query = self.transformer.tgt_embed.weight[None].repeat(N, 1, 1)
        else:
            query = target_unact

        # query = torch.zeros_like(query)
        # query  = torch.randn((N, self.num_queries, self.embed_dim), device=self.device, dtype=torch.float)
        t = torch.full((N,),self.num_timesteps - 1, device=self.device, dtype=torch.long)
        _ , reference_points = self.apply_box_noise(init_reference , t = t)
        # reference_points = query[:,:,-4:]
        # init_reference_points = reference_points.float()
        
        noise = torch.clamp(reference_points, min=-1 * self.scale, max=self.scale)
        noise = ((noise / self.scale) + 1) / 2
        # reference_points = query[:,:,-4:]
        init_reference_points = noise.float()
        ensemble_score, ensemble_label, ensemble_coord = [], [], []

        for time, time_next in time_pairs:
            # index = self.num_timesteps - time - 1

            t = torch.full((N,),time, device=self.device, dtype=torch.long)
            time_cond = timestep_embedding(t.float() , self.transformer.embed_dim, repeat_only=False)
            time_cond = self.transformer.time_embed(time_cond)
            # self_cond = x_start if self.self_condition else None
            (inter_states,
            inter_references)= self.transformer.decoder(
                query = query,  # bs, num_queries, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=init_reference_points,  # num_queries, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                t = time_cond,
            )
            outputs_classes = []
            outputs_coords = []

            for lvl in range(inter_states.shape[0]):
                # current_inter_states = inter_states[lvl]
                # class_state , ref_points_state = current_inter_states.chunk(2, dim = -1)
                if lvl == 0:
                    reference = init_reference_points
                else:
                    reference = inter_references[lvl - 1]
                outputs_class = self.class_embed[lvl](inter_states[lvl])
                tmp = self.bbox_embed[lvl](inter_states[lvl])
                # assert reference.shape[-1] == 4
                if reference.shape[-1] == 4:
                    tmp += inverse_sigmoid(reference)
                    # tmp = tmp
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
        
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            # select parameters corresponding to the currently considered timestep
            # bbox_start = inverse_sigmoid(outputs_coord[-1])
            # bbox_start = bbox_start.clamp(min = -4 , max = 4)
            bbox_start = outputs_coord[-1]
            x_start = (bbox_start * 2 - 1.) * self.scale
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

            # init_reference_points = inverse_sigmoid(init_reference_points)
            
            # bbox_start = bbox_start.clamp(min = -3 ,max = 3)
            # init_reference_points =init_reference_points.clamp(min = -3 ,max = 3)

            # query_start = inter_states[-1]
            # init_reference_points = init_reference_points
            # bbox_start = outputs_coord[-1]

            # pred_noise_query = self.predict_noise_from_start(query_start, t , query)
            pred_noise_bbox = self.predict_noise_from_start(reference_points, t , x_start)

            # pred_noise_query = self.predict_noise_from_start(query, t , query_start)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # query = query_start * alpha_next.sqrt() + \
            #       c * pred_noise_query 
        
            reference_points = bbox_start * alpha_next.sqrt() + \
                  c * pred_noise_bbox 
            noise = torch.clamp(reference_points, min=-1 * self.scale, max=self.scale)
            noise = ((noise / self.scale) + 1) / 2
            # reference_points = query[:,:,-4:]
            init_reference_points = noise.float()
            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1], outputs_coord[-1]
                threshold = 0.5 #0.5

                ##adaptive threshold
                threshold = time_next/100 ##v1
                # threshold = (time+time_next)/200 #v2
                assert threshold>=0 and threshold<=1, f"{threshold} fix threshold"
                # print(threshold)

                score_per_image = score_per_image.sigmoid()
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold 
                num_remain = torch.sum(keep_idx)
                new_ref = torch.randn_like(init_reference_points , device = init_reference_points.device)
                reference = new_ref
                reference[keep_idx] = reference_points[keep_idx].float()
                reference_points = reference
                new_ref = torch.clamp(new_ref, min=-1 * self.scale, max=self.scale)
                new_ref = ((new_ref / self.scale) + 1) / 2
                # pred_logits =  torch.cat((pred_logits , outputs_class[-1][keep_idx]), dim=1)
                # pred_boxes =  torch.cat((pred_boxes , outputs_coord[-1][keep_idx] ), dim=1)
                new_ref[keep_idx] = init_reference_points[keep_idx] 
                init_reference_points = new_ref
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes, final = False)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)

            ## no ensemble but with nms
            # box_pred_per_image = ensemble_coord[-1]
            # scores_per_image = ensemble_score[-1]
            # labels_per_image = ensemble_label[-1]
            
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.8) #0.5
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
            # reference_points = query[:,:,-4:]
            # init_reference_points = reference_points.clamp(min = -3 , max = 3)
            # init_reference_points = reference_points.float().sigmoid()
            # init_reference_points = reference_points.float()

            # t1 = torch.full((N,),time_next, device=self.device, dtype=torch.long)

            # # _ , reference_points = self.apply_box_noise(init_reference , t = t)

            # reference_points = self.q_sample(x_start = bbox_start , t = t1, noise = None)
            
            # # reference_points = query[:,:,-4:]
            # init_reference_points = reference_points.float().sigmoid()
            # s = torch.randperm(init_reference_points.size(1))
            # # Shuffle the tensor
            # init_reference_points = init_reference_points[:,s,:]

            # query = query.float()
        # outputs_class = self.class_embed[-1](query)
        # outputs_coord = self.bbox_embed[-1](query)
            sorted_indices = torch.argsort(scores_per_image, descending=True)
            top_k = self.select_box_nums_for_evaluation
            selected_indices = sorted_indices[:top_k]
            # Extract top results
            box_pred_per_image = box_pred_per_image[selected_indices]
            scores_per_image = scores_per_image[selected_indices]
            labels_per_image = labels_per_image[selected_indices]
            image_size = images.image_sizes[0]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    def prepare_for_diffusion(self, gt_boxes, labels, num_gt, is_train):
        t = torch.randint(0,self.num_timesteps,(1,), device=self.device).long() 
        # no_object_label = self.label_enc(torch.tensor(self.num_classes, dtype=torch.long , device=self.device))
        # no_object_bbox = torch.tensor([0,0,0,0], device = self.device) 
        # num_gt = len(gt_boxes) # number of objects
        # no_object_label = no_object_label.repeat(self.num_queries - num_gt, 1) # 295 , 256
        # no_object_bbox = no_object_bbox.repeat(self.num_queries - num_gt , 1) # 295 , 4

        # labels = torch.cat([labels, no_object_label] ,dim = 0) # 900 , 256
        # gt_boxes = torch.cat([gt_boxes, no_object_bbox] , dim = 0) # 900 , 4

        queries = torch.cat([labels , gt_boxes] , dim = 1)

        # noise_bbox = torch.randn(gt_boxes.shape, device=self.device) 
        noise_labels = torch.randn(labels.shape, device=self.device)

        # gt_boxes = (gt_boxes * 2. - 1.) * self.box_noise_scale
        # init_anchors = self.prior_anchors.weight
        # noise sample 
        diff_boxes , noise = self.apply_box_noise(gt_boxes , t)
        # diff_boxes = diff_boxes[..., :2]
        # diff_boxes = self.q_sample(x_start=gt_boxes, t=t, noise=None)
        # diff_boxes = torch.clamp(diff_boxes, min= -1 * self.box_noise_scale, max=self.box_noise_scale)
        # diff_boxes = ((diff_boxes / self.box_noise_scale) + 1) / 2.
        # labels = (labels * 2. - 1.) * self.label_noise_ratio
        diff_labels = self.q_sample(x_start = labels , t = t, noise = noise_labels)
        # labels = torch.clamp(labels,min = -1 * self.label_noise_ratio , max = self.label_noise_ratio)
        # diff_labels = ((labels / self.label_noise_ratio) + 1) / 2.
        noisy_queries = torch.cat([diff_labels , diff_boxes] , dim = 1)

        shuffled_indices = torch.randperm(diff_labels.size(0))
        # Shuffle the tensor
        diff_labels = diff_labels[shuffled_indices]
        diff_boxes = diff_boxes[shuffled_indices]
        shuffled_indices = torch.argsort(shuffled_indices)
        # Create tuples pairing old indices with new shuffled indices
        indices = shuffled_indices[:num_gt] , torch.arange(diff_labels.size(0))[:num_gt]
        # indices = torch.arange(diff_labels.size(0))[:num_gt] , torch.arange(diff_labels.size(0))[:num_gt]
        # indices = None
        return noisy_queries, diff_boxes, queries, t, indices


    def process_targets(self, targets, new_targets_diffusion, is_train = True):
        init_query_bboxes , noisy_queries, original_queries, ts, indices, num_gts = [], [], [] , [], [], []
        for  target, target_diffusion in zip(targets, new_targets_diffusion):
            # bbox = [bbox for bbox in target['boxes']]
            # labels = [bbox for bbox in target['labels']]
            bbox , labels = target_diffusion['boxes'] , target_diffusion['labels']
            num_gt = len(target['boxes'])
            labels = self.label_enc(labels.squeeze())   # number of object , 256
            noisy_query, diff_boxes, queries, t, indice = self.prepare_for_diffusion(bbox, labels, num_gt, is_train=is_train)
            init_query_bboxes.append(diff_boxes.unsqueeze(0))
            noisy_queries.append(noisy_query.unsqueeze(0))
            original_queries.append(queries.unsqueeze(0))
            num_gts.append(num_gt)
            ts.append(t)
            indices.append(indice)
        return (torch.cat(noisy_queries,dim = 0), torch.cat(init_query_bboxes,dim=0)
                ,torch.cat(original_queries,dim = 0),torch.cat(ts,dim=0)
                ,indices, num_gts) 