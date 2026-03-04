import torch 
import torch.nn as nn
from .mlp import MLP
from .ldm.modules.diffusionmodules.util import (
    linear,
    conv_nd,
    zero_module,

)
import einops

def exists(x):
    return x is not None


class TimeStepBlock(nn.Module):
    def __init__(self,
                 channels,
                 emb_channels,
                 out_channels = 256,
                 dims = 1,
                 dropout = 0.2,
                 use_scale_shift_norm = True):
        super(TimeStepBlock, self).__init__()
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        # self.out_norm = nn.LayerNorm(self.out_channels)
        # self.out_norm = nn.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6)
        # self.out_layers = nn.Sequential(
        #     # nn.LayerNorm(self.out_channels),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     zero_module(
        #         conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        #     ),
        # )

    def forward(self, x, time_embed):
        emb_out = self.emb_layers(time_embed).type(x.dtype)

        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[:, None]

        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=-1)
            h = x * (1 + scale) + shift
            # h = self.out_norm(h)
        return x + h

class BBoxEmbed(nn.Module):
    def __init__(self,embed_dim,
                time_embed_channels,):
        super(BBoxEmbed, self).__init__()

        self.pred = MLP(embed_dim, embed_dim, 4, 3)
        self.norm = nn.LayerNorm(embed_dim)
        # self.time_embedding_layer = 
        self.time_step_embed = TimeStepBlock(
                    channels = embed_dim,
                    emb_channels = time_embed_channels,
                    # channels = 256,
                )
    def forward(self , x , time_embed = None):
        # x = self.time_step_embed(x , time_embed)
        # x = self.pred(self.norm(x))
        x = self.pred(x)
        return x
    
class ClassEmbed(nn.Module):
    def __init__(self,embed_dim,
                time_embed_channels,
                num_classes):
        super(ClassEmbed , self).__init__()
        # self.attention = CrossAttention(query_dim=embed_dim, context_dim=context_dim,
        #                             heads=n_heads, dim_head=d_head, dropout=dropout)
        self.num_classes = num_classes
        self.pred = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
                    channels = embed_dim,
                    emb_channels = time_embed_channels,
                    # channels = 256,
                )

    def forward(self , x , time_embed=None):
        # x = self.time_step_embed(x , time_embed)
        # x = self.pred(self.norm(x))
        x = self.pred(x)
        return x

