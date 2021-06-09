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
                 augmix_used=False,
                 aug_split_num=None):
        super(DynamicImageClassifier,self).__init__(
                backbone,
                neck=neck,
                head=head,
                pretrained=pretrained,
                train_cfg=train_cfg)

        self.augmix_used=augmix_used
        self.aug_split_num=aug_split_num
        
        if self.augmix_used:
            assert aug_split_num is not None

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)

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
        
        if(self.augmix_used):
            # to do: label 感觉也得处理成对应的shape，否则下面的mixup不兼容, 因为JSDloss是跟下面这个处理对齐的，所以JSDloss也得改（）
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
        
        loss = self.head.forward_train(x, gt_label, **kwargs)
        
        losses.update(loss)

        return losses


    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x, **kwargs)



    def forward_test(self, imgs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def forward(self, img, return_loss=True, **kwargs):
        # Not any change in this function, just make a expression
        # 在调用model(return_logits=True)的时候，会返回logits，
        # 本来返回的是score
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        #import pdb
        #pdb.set_trace()
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def train_step(self, data, optimizer, teacher_logits=None, **kwargs):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        
        losses = self(**data, teacher_logits=teacher_logits, **kwargs)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data),logits=losses.get('logits',None))

        return outputs

        
    def forward_dummy(self, img):
        # needn't img_metas, just use None to placeholder
        logit = self.forward_test(img, return_logits=True, img_metas=None)
        return logit