import copy
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
from detrex.config import get_config
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.config import LazyConfig

#v3det_dino_r50_4scale_40ep_better_hyper_badran_exp17_repeatsampler_bs8

train.init_checkpoint = "../pretrained/r50.pkl"

train.output_dir = "../exps/diffu_dino_24ep_r50_v3det"

model.dino_type = "diffu_det"


dataloader.evaluator.output_dir = train.output_dir + '/inference'
train.model_ema.enabled=True
train.model_ema.use_ema_weights_for_eval_only=True

model.noisy_gt = True
model.criterion.denoise_noobject_bbox = False
model.criterion.sigmoid_focal_loss = True

model.old_schedule = True
model.criterion.use_vlb = False

model.transformer.learnt_init_query = False
model.transformer.tgt_embed_new = False ## if learnt_init_query=False then tgt_embed_new is useless

model.criterion.use_matcher = True
model.criterion.use_encoder_loss = True

model.timesteps = 100
model.sampling_steps = 1
model.aux_loss = True
model.num_queries = 900

model.criterion.use_count_loss = False

####
train.ddp.find_unused_parameters= True


# no frozen backbone get better results
model.backbone.freeze_at = -1

# more dn queries, set 300 here
model.dn_number = 300

# use 2.0 for class weight
model.criterion.weight_dict = {
    "loss_class": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_class_dn": 1,
    "loss_bbox_dn": 5.0,
    "loss_giou_dn": 2.0,
}


# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict


dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16 #32 4gpu
dataloader.test.batch_size = 16 #32 4gpu
dataloader.test.num_workers = 4

dataloader = LazyConfig.load("../datasets_utils/v3detr_detr.py").dataloader
lr_multiplier = LazyConfig.load("../datasets_utils/v3det_schedule.py").lr_multiplier_vit3det24ep_bs8
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8 #32 4gpu

# max training iterations
train.max_iter = 275000*2
# run evaluation every 5000 iters
train.eval_period = 68750*4*2
# log training infomation every 20 iters
train.log_period = 40
# save checkpoint every 5000 iters
train.checkpointer.period = 5000

dataloader.evaluator.output_dir = train.output_dir + '/inference'


from detrex.modeling.matcher import HungarianMatcher
from projects.dino.modeling import    DINOCriterion

model.num_classes=13204
model.criterion.num_classes=13204 


dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)

# train.init_checkpoint = train.output_dir + "/model_final.pth"
model.sampling_steps = 3
dataloader.evaluator.output_dir = train.output_dir + '/inference3steps'
dataloader.test.batch_size = 1 