# third lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead

# gaia lib
from gaiavision.core.ops import DynamicLinear

# local lib
from ..losses.accuracy import Accuracy

@HEADS.register_module()
class DynamicLinearClsHead(LinearClsHead):
    """Dynamic Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map(dynamic).
    """
   
    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super(DynamicLinearClsHead, self).__init__(num_classes,in_channels,
                                                   *args, **kwargs)
        
        self._init_layers()
        self.compute_accuracy = Accuracy(topk=self.topk)

    def _init_layers(self):
        # Just change this nn.Linear -> DynamicLinear
        self.fc = DynamicLinear(self.in_channels, self.num_classes)

    # 因为MixAug会导致cls_score的shape和gt_label的shape不匹配，所以要重新做些小修改
    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses   
