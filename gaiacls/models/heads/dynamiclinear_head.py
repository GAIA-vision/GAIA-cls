import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead

from gaiavision.core.ops import DynamicLinear


@HEADS.register_module()
class DynamicLinearClsHead(LinearClsHead):
    """Dynamic Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map(dynamic).
    """
    # 这样是不行的。。。 是不是应该继承LinearClsHead,而不是ClsHead,好像是这样 2333，组会开完改一下试试
    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super(DynamicLinearClsHead, self).__init__(num_classes,in_channels,
                                                   *args, **kwargs)
        
        self._init_layers()

    def _init_layers(self):
        # Just change this nn.Linear -> DynamicLinear
        self.fc = DynamicLinear(self.in_channels, self.num_classes)


