# refer from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py

import copy
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from mmcls.datasets.pipelines import Compose


# 这个为了兼容到mmcls里面还得仔细构思一下，看下哪些接口需要更改。
# 这个dataset可能不一定会用的到，这其中的很多操作可能直接在进行Aug的pipeline里面
# 进行实现比较好。因为仔细思考了下，这个dataset本身就包含了Aug操作，但是mm的框架中是
# 把加载数据和Aug数据是分开了的，所以这里可能还是要分开。
# 这个待定，还得从实现方式的难易程度上进行具体的判断（）
class AugMixDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    CLASSES=None
    
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits
    
    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    # 这块保持不动，或者和torch.utils.data.Dataset保持一致也没关系，仔细思考了下，
    # 看实现方式上能否通过在配置文件上进行操作，就是让pipelines输入的时候是一个列表，
    # 然后在这个AugMixDataset的 self.prepare_data中修改一下self.pipeline的使用方式。
    # 实现的时候最后别忘再注意下是否会对后续的loss计算产生干扰，影响后面loss模块的代码。
    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
