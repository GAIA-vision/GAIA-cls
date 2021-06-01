import pdb

import torch

# mm lib
from mmcls.models import CLASSIFIERS, ImageClassifier

# gaia lib
from gaiavision.core import DynamicMixin


@CLASSIFIERS.register_module()
class DynamicImageClassifier(ImageClassifier, DynamicMixin):
    
    search_space = {'backbone'}

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 aug_mix_used=False,
                 aug_split_num=None):
        super(DynamicImageClassifier,self).__init__(
                backbone,
                neck=neck,
                head=head,
                pretrained=pretrained,
                train_cfg=train_cfg)

        self.aug_mix_used=aug_mix_used
        self.aug_split_num=aug_split_num
        
        if self.aug_mix_used:
            assert aug_split_num is not None

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)


    # 这个需要重写，应对AugMix
    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        if(self.aug_mix_used):
            
            batch_size,C,H,W = img.shape
            original_channel_num = C // self.aug_split_num
            assert original_channel_num * self.aug_split_num == C

            targets = torch.zeros((batch_size*self.aug_split_num, original_channel_num,H,W),
                                  device=img.device,
                                  dtype=img.dtype)
            for i in range(batch_size):
                for j in range(self.aug_split_num):
                    targets[i + j * batch_size] = img[i][j*original_channel_num:(j+1)*original_channel_num,:,:]
            # pdb.set_trace()
            img = targets
            
        if self.mixup is not None:
            img, gt_label = self.mixup(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses
