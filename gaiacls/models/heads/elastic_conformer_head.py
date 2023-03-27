from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, constant_init, kaiming_init

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads import ClsHead
from gaiavision.core.ops import ElasticLinear
from gaiavision.core import DynamicMixin
from gaiavision.core.bricks import build_norm_layer

@HEADS.register_module()
class ElasticConformerClsHead(ClsHead, DynamicMixin):
    def __init__(self,
                 num_classes,
                 in_channels,
                 channel_ratio,
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                 *args,
                 **kwargs):
        super(ElasticConformerClsHead, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channel_ratio = channel_ratio

        self.trans_cls_head = ElasticLinear(self.in_channels, self.num_classes)
        self.conv_cls_head = ElasticLinear(int(self.channel_ratio), self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.in_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.compute_loss = build_loss(loss)
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def loss(self, conv_cls, tran_cls, gt_label):
        num_samples = len(conv_cls)
        losses = dict()
        loss1 = self.compute_loss(conv_cls, gt_label, avg_factor=num_samples)
        loss2 = self.compute_loss(tran_cls, gt_label, avg_factor=num_samples)
        
        loss = (loss1 + loss2) / 2
        losses['loss'] = loss

        return losses
    
    def simple_test(self, x, x_t, **kwargs):
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        x_t = self.norm1(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        cls_score = conv_cls + tran_cls
        if kwargs.get('return_logits', False):
            return cls_score
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        on_trace = hasattr(torch.jit, 'is_tracing') and torch.jit.is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, x_t, gt_label, **kwargs):
        # conv classification
        x_p = self.pooling(x).flatten(1) # bs 256 1 1 --> bs 256
        conv_cls = self.conv_cls_head(x_p) # bs 1000
        # trans classification
        x_t = self.norm1(x_t) # bs 197 384
        tran_cls = self.trans_cls_head(x_t[:, 0]) # bs 1000

        losses = self.loss(conv_cls, tran_cls, gt_label)
        return losses
