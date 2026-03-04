from detrex.config import get_config
from ..models.dino_swin_base_384 import model
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train
from detectron2.data.samplers import RepeatFactorTrainingSampler


# modify training config
train.init_checkpoint = "../pretrained/swin_base_patch4_window12_384_22kto1k.pkl"

train.output_dir = "../exps/diffu_dino_12ep_swin_v3det"

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
####
train.ddp.find_unused_parameters= True


# max training iterations
train.max_iter = 180000
train.eval_period = 5000*2
train.log_period = 20*2
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1


dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16 
dataloader.test.batch_size = 16 
dataloader.test.num_workers = 4


dataloader = LazyConfig.load("../datasets_utils/v3detr_detr.py").dataloader
lr_multiplier = LazyConfig.load("../datasets_utils/v3det_schedule.py").lr_multiplier_vit3det24ep_bs8
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8 

# max training iterations
train.max_iter = 275000*2
train.eval_period = 68750*4*2
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


model.sampling_steps = 3
dataloader.evaluator.output_dir = train.output_dir + '/inference3steps'
dataloader.test.batch_size = 1 
