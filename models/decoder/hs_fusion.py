# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops

import copy

# 已删除DCNv2文件夹，使用torchvision.ops替代
# from dcn_v2 import DCN as dcn_v2
from torchvision.ops import DeformConv2d as dcn_v2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HsFusion(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone_channels, hidden_dim, num_feature_levels, with_fpn=False, method_fpn=""):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            with_fpn: 是否使用了FPN来进行特征融合
            method_fpn: 用了什么方式来融合(fpn, bifpn, fapn, pafpn)
        """
        super().__init__()

        self.num_feature_levels = num_feature_levels
        self.with_fpn = with_fpn
        self.method_fpn = method_fpn

        # 多尺度的特征图输入
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            if not self.with_fpn:
                for _ in range(num_feature_levels - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                    in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        if self.with_fpn:
            in_channel = backbone_channels[-1]
            # 获得最高层的feature map (这里都是默认为向上提取一层)
            self.top_feature_proj = nn.ModuleList()
            for i in range(num_feature_levels - num_backbone_outs):
                self.top_feature_proj.append(nn.Sequential(
                    nn.Conv2d(in_channel, in_channel // 2, kernel_size=1),
                    nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channel // 2, in_channel * 2, kernel_size=1),
                    nn.GroupNorm(32, in_channel * 2),
                ))
            # proj list
            self.fpn_proj_list = nn.ModuleList()
            for _ in range(num_feature_levels):
                self.fpn_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

            if self.method_fpn == "fpn":
                self.bottomup_conv1 = nn.ModuleList()
                for _ in range(num_feature_levels - 2, -1, -1):
                    in_channels = backbone_channels[_]
                    self.bottomup_conv1.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel)
                    ))

            if self.method_fpn == "pafpn":
                self.bottomup_conv = nn.ModuleList()
                self.upbottom_conv = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone_channels[num_feature_levels - i - 2]
                    self.upbottom_conv.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel),
                    ))
                    self.bottomup_conv.append(nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, in_channel),
                    ))

                for c1, c2 in zip(self.bottomup_conv, self.upbottom_conv):
                    nn.init.xavier_uniform_(c1[0].weight, gain=1)
                    nn.init.constant_(c1[0].bias, 0)
                    nn.init.xavier_uniform_(c2[0].weight, gain=1)
                    nn.init.constant_(c2[0].bias, 0)

            if self.method_fpn == "bifpn":
                self.upbottom_conv = nn.ModuleList()
                self.bottomup_conv1 = nn.ModuleList()
                self.bottomup_conv2 = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone_channels[num_feature_levels - i - 2]
                    if i + 1 == num_feature_levels - 1:
                        in_channels_cross = in_channel * 2
                    else:
                        in_channels_cross = backbone_channels[i + 1]
                    self.upbottom_conv.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel),
                    ))
                    self.bottomup_conv1.append(nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, in_channel),
                        nn.ReLU(inplace=True)
                    ))
                    self.bottomup_conv2.append(nn.Sequential(
                        nn.Conv2d(in_channels_cross, in_channel, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(32, in_channel),
                        nn.ReLU(inplace=True)
                    ))

                for c1, c2, c3 in zip(self.upbottom_conv, self.bottomup_conv1, self.bottomup_conv2):
                    nn.init.xavier_uniform_(c1[0].weight, gain=1)
                    nn.init.constant_(c1[0].bias, 0)
                    nn.init.xavier_uniform_(c2[0].weight, gain=1)
                    nn.init.constant_(c2[0].bias, 0)
                    nn.init.xavier_uniform_(c3[0].weight, gain=1)
                    nn.init.constant_(c3[0].bias, 0)

            if self.method_fpn == "fapn":
                self.align_modules = nn.ModuleList()
                self.bottomup_conv = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone_channels[num_feature_levels - i - 2]
                    align_module = FeatureAlign_V2(in_channels, hidden_dim)
                    self.align_modules.append(align_module)
                for i in range(num_feature_levels):
                    self.bottomup_conv.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU()
                    ))

            if self.method_fpn == "wbcfpn":
                self.upbottom_conv = nn.ModuleList()
                self.weight_conv = nn.ModuleList()
                self.lateral_conv = nn.ModuleList()
                # self.bottomup_lateral_conv = nn.ModuleList()
                # self.up_conv = nn.ModuleList()
                # self.cbam_attention = nn.ModuleList()
                self.up_sample_conv = nn.ModuleList()
                for i in range(num_feature_levels):
                    if i == 0:
                        in_channels = in_channel * 2
                    else:
                        in_channels = backbone_channels[num_feature_levels - i - 1]
                    self.upbottom_conv.append(nn.Sequential(
                        ChannelAttention(in_channels),
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                    ))
                    if i != 0:
                        self.weight_conv.append(nn.Sequential(
                            ChannelAttention(hidden_dim, ratio=4, flag=False)
                        ))
                        self.up_sample_conv.append(nn.Sequential(
                            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2,
                                               padding=1, output_padding=1),
                        ))
                    self.lateral_conv.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU(),
                    ))

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def _upsample_add(self, x, y):
        """
        上采样并且将两个feature map进行相加
        Parameters:
            x: 上层的feature map
            y: 融合的feature map
        """
        _, _, h, w = y.size()
        return F.upsample(x, size=(h, w), mode='bilinear') + y

    def fpn(self, srcs):
        """
        最普通的FPN方式，通过element wise方式将不同层的feature map进行相加
        Parameters:
            srcs:不同层的feature map
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        results = [feature_maps[0]]
        prev_feature = feature_maps[0]
        for feature, conv in zip(feature_maps[1:], self.bottomup_conv1):
            prev_feature = self._upsample_add(prev_feature, conv(feature))
            results.insert(0, prev_feature)
        return results

    def pafpn(self, srcs):
        """
        这是PANet中特征融合的方式：
        paper: https://arxiv.org/abs/1803.01534
        code: https://github.com/ShuLiu1993/PANet
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        up_bottom_features = []
        bottom_up_features = []
        up_bottom_features.append(feature_maps[0])
        # 从上到下的特征融合
        for feature, conv in zip(feature_maps[1:], self.upbottom_conv):
            prev_feature = self._upsample_add(up_bottom_features[0], conv(feature))
            up_bottom_features.insert(0, prev_feature)

        bottom_up_features.append(up_bottom_features[0])
        for i in range(1, len(up_bottom_features)):
            prev_feature = self.bottomup_conv[i - 1](bottom_up_features[0])
            prev_feature = prev_feature + up_bottom_features[i]
            bottom_up_features.insert(0, prev_feature)

        return bottom_up_features[::-1]

    def bifpn(self, srcs):
        """
        这是EfficientDet的特征融合方式:
        paper: https://arxiv.org/abs/1911.09070
        code: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        up_bottom_features = []
        bottom_up_features = []
        up_bottom_features.append(feature_maps[0])
        for feature, conv in zip(feature_maps[1:], self.upbottom_conv):
            prev_feature = self._upsample_add(up_bottom_features[0], conv(feature))
            up_bottom_features.insert(0, prev_feature)

        bottom_up_features.append(up_bottom_features[0])
        for i in range(1, len(up_bottom_features)):
            prev_feature = self.bottomup_conv1[i - 1](bottom_up_features[0])
            prev_feature = prev_feature + up_bottom_features[i] + \
                           self.bottomup_conv2[i - 1](srcs[i])
            bottom_up_features.insert(0, prev_feature)

        return bottom_up_features[::-1]

    def fapn(self, srcs):
        """
        这是FaPN的特征融合:
        paper:
        code:https://github.com/ShihuaHuang95/FaPN-full
        """
        # reverse feature map
        feature_maps = srcs[::-1]
        results = [feature_maps[0]]
        for feature, align_module in zip(feature_maps[1:], self.align_modules):
            prev_feature = align_module(feature, results[0])
            results.insert(0, prev_feature)

        for i in range(self.num_feature_levels):
            results[i] = self.bottomup_conv[i](results[i])

        return results

    def wbcfpn(self, srcs):
        """
        这是我们自己设计的FPN
        """
        feature_maps = srcs[::-1]
        up_sample_features = []
        up_bottom_features = []
        # bottom_up_features = []
        up_bottom_features.append(self.upbottom_conv[0](feature_maps[0]))
        up_sample_features.append(up_bottom_features[0])
        for up_sample in self.up_sample_conv:
            up_sample_features.insert(0, up_sample(up_sample_features[0]))

        up_sample_features = up_sample_features[::-1]

        for i, (feature, conv, weight_conv) in enumerate(
                zip(feature_maps[1:], self.upbottom_conv[1:], self.weight_conv)):
            down_feature = conv(feature)
            _, _, h, w = feature.shape
            high_feature = up_sample_features[i + 1]
            if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
                high_feature = F.upsample(high_feature, size=(h, w), mode='bilinear')

            select_down_feature = weight_conv(high_feature) * down_feature
            fusion_feature = select_down_feature + high_feature
            up_bottom_features.append(fusion_feature)

        results = up_bottom_features[::-1]
        for i in range(len(results)):
            results[i] = self.lateral_conv[i](results[i])

        return results

    def get_fpn(self, method_fpn, srcs):
        """
        Parameters:
            method_fpn: fpn的方式
            srcs: 输入的特征图
        """
        fpn_map = {
            'fpn': self.fpn,
            'bifpn': self.bifpn,
            'pafpn': self.pafpn,
            'fapn': self.fapn,
            'wbcfpn': self.wbcfpn
        }
        assert method_fpn in fpn_map, f'do you really want to using the {method_fpn} ?'
        return fpn_map[method_fpn](srcs)

    def forward(self, features):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        srcs = features
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if self.with_fpn:
                    if l == _len_srcs:
                        src = self.top_feature_proj[_len_srcs - l](features[-1])
                    else:
                        src = self.top_feature_proj[_len_srcs - l](srcs[-1])
                    srcs.append(src)
                else:
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1])
                    else:
                        src = self.input_proj[l](srcs[-1])

                    srcs.append(src)

        # 使用FPN来进行融合
        if self.with_fpn:
            srcs = self.get_fpn(self.method_fpn, srcs)
            if self.method_fpn != "fapn" and self.method_fpn != "wbcfpn":
                for i in range(len(srcs)):
                    srcs[i] = self.fpn_proj_list[i](srcs[i])

        else:
            for i in range(len(srcs) - 1):
                srcs[i] = self.input_proj[i](srcs[i])

        return srcs


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x if self.flag else self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, flag=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x if self.flag else self.sigmoid(out)


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(32, in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(32, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.group_norm1 = nn.GroupNorm(32, out_nc)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.offset.weight)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.group_norm1(
            self.offset(torch.cat([feat_arm, feat_up * 2], dim=1)))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm
