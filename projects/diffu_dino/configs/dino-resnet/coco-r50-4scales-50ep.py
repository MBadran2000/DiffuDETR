import copy
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
from detrex.config import get_config


lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
train.max_iter = 375000
train.eval_period = 5000*4

train.init_checkpoint = "../pretrained/r50.pkl"

train.output_dir = "../exps/diffu_dino_50ep_r50"

model.dino_type = "diffu_det"
model.beta_schedule = "cosine"
model.old_schedule = False


dataloader.evaluator.output_dir = train.output_dir + '/inference'
train.model_ema.enabled=True
train.model_ema.use_ema_weights_for_eval_only=True

model.noisy_gt = True
model.criterion.denoise_noobject_bbox = False
model.criterion.sigmoid_focal_loss = True

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


dataloader.train.num_workers = 40
dataloader.train.total_batch_size = 16 
dataloader.test.batch_size = 16 
dataloader.test.num_workers = 40



model.sampling_steps = 3
dataloader.evaluator.output_dir = train.output_dir + '/inference3steps'
dataloader.test.batch_size = 1 