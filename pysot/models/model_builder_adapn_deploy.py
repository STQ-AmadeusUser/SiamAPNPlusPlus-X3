# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config_adapn import cfg
from pysot.models.backbone.alexnet import AlexNet
from pysot.models.utile_adapn_deploy import APN, clsandloc


class ModelBuilderADAPN(nn.Module):
    def __init__(self, device=None):
        super(ModelBuilderADAPN, self).__init__()

        self.backbone = AlexNet()
        self.grader = APN(cfg)
        self.new = clsandloc(cfg)
        if device is not None:
            self.backbone = self.backbone.to(device)
            self.grader = self.grader.to(device)
            self.new = self.new.to(device)

    def forward(self, template, search):
        """ only used in training
        """

        zf = self.backbone(template)  # 1x384x8x8, 1x256x6x6
        xf = self.backbone(search)  # 1x384x28x28, 1x256x26x26

        xff, ress = self.grader(xf, zf)  # 1x4x21x21,  1x256x21x21

        cls1, cls2, cls3, loc = self.new(xf, zf, ress)
        # cls1: 1x2x21x21
        # cls2: 1x2x21x21
        # cls3: 1x1x21x21
        # loc: 1x4x21x21

        cls1 = cls1.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # 441x2
        cls1 = F.softmax(cls1, dim=1)[:, 1]  # (441,)
        cls2 = cls2.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # 441x2
        cls2 = F.softmax(cls2, dim=1)[:, 1]  # (441,)
        cls3 = cls3.contiguous().view(-1)  # (441,)

        return loc, cls1, cls2, cls3, xff

    def template(self, z):

        zf = self.backbone(z)
        self.zf = zf

    def track(self, x):

        xf = self.backbone(x)
        xff, ress = self.grader(xf, self.zf)

        self.ranchors = xff

        cls1, cls2, cls3, loc = self.new(xf, self.zf, ress)

        return {
            'cls1': cls1,
            'cls2': cls2,
            'cls3': cls3,
            'loc': loc
        }
