import torch

# mm lib
from mmcls.models import CLASSIFIERS, ImageClassifier
from mmcls.models.utils import Augments
# gaia lib
from gaiavision.core import DynamicMixin

@CLASSIFIERS.register_module()
class ElasticeImageClassifierConformer(ImageClassifier, DynamicMixin):
    search_space = {'backbone'}
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None):
        super(ElasticeImageClassifierConformer,self).__init__(
                backbone,
                neck=neck,
                head=head,
                pretrained=pretrained,
                train_cfg=train_cfg)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                print("-------------------------Augments used!-------------------------")
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)
        
    def forward_train(self, img, gt_label, **kwargs):
        if self.augments is not None:
            img, gt_label= self.augments(img, gt_label)

        x, x_t = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, x_t, gt_label, **kwargs)
        losses.update(loss)
        return losses

    def forward_test(self, imgs, **kwargs):
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def simple_test(self, img, img_metas, **kwargs):
        x, x_t = self.extract_feat(img)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        return self.head.simple_test(x, x_t, **kwargs)
    
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)
    def forward_dummy(self, img):
        logit = self.forward_test(img, return_logits=True, img_metas=None)
        return logit