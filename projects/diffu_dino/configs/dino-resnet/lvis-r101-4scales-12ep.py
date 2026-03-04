from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
from detrex.config import get_config

train.init_checkpoint = "../pretrained/r101.pkl"

train.output_dir = "../exps/diffu_dino_12ep_r101_lvis"

# modify model config
model.backbone.stages.depth = 101


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
model.transformer.tgt_embed_new = False 

model.criterion.use_matcher = True
model.criterion.use_encoder_loss = True

model.timesteps = 100
model.sampling_steps = 1
model.aux_loss = True
model.num_queries = 900

model.criterion.use_count_loss = False


train.ddp.find_unused_parameters= True


dataloader.train.num_workers = 40
dataloader.train.total_batch_size = 16 
dataloader.test.batch_size = 16
dataloader.test.num_workers = 40

dataloader = get_config("../datasets_utils/lvis_detr_diffusion.py").dataloader
lr_multiplier = get_config("../datasets_utils/lvis_schedule.py").lr_multiplier_12ep
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16

# max training iterations
train.max_iter = 75000
train.eval_period = 25000
# log training infomation every 20 iters
train.log_period = 20
# save checkpoint every 5000 iters
train.checkpointer.period = 5000

dataloader.evaluator.output_dir = train.output_dir + '/inference'

from detrex.modeling.matcher import HungarianMatcher
from projects.dino.modeling import    DINOCriterion

model.num_classes=1203 
model.criterion.num_classes=1203 


# train.init_checkpoint = train.output_dir + "/model_final.pth"
model.sampling_steps = 3
dataloader.evaluator.output_dir = train.output_dir + '/inference3steps'
dataloader.test.batch_size = 1 