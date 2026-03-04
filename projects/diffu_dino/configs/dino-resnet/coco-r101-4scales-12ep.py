from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "../pretrained/r101.pkl"

train.output_dir = "../exps/diffu_dino_12ep_r101"

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

####
train.ddp.find_unused_parameters= True


model.sampling_steps = 3
dataloader.evaluator.output_dir = train.output_dir + '/inference3steps'
dataloader.test.batch_size = 1 