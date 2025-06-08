"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


# from hybrid_encoder import HybridEncoder
# from rtdetr_decoder import RTDETRTransformer


# from src.core import register


# __all__ = ['RTDETR', ]


# @register
class RTDETR(nn.Module):
    # __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone, encoder, decoder, fusion=None, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        # self.fusion = fusion
        self.encoder = encoder
        self.decoder = decoder

        self.multi_scale = multi_scale

    def forward(self, x, query_pos, ref_points, targets=None):

        # todo 后续更改
        # if self.multi_scale and self.training:
        #     sz = np.random.choice(self.multi_scale)
        #     x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        # x = self.fusion(x)[1:]
        x = self.encoder(x)
        out_logits, out_bboxes, output_last, inter_ref_bbox, query_pos_embed, init_query = self.decoder(x, query_pos,
                                                                                                        ref_points,
                                                                                                        targets)

        return out_logits, out_bboxes, output_last, inter_ref_bbox, query_pos_embed, init_query

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
