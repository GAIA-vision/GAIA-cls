# mm lib
from mmcls.models import CLASSIFIERS, ImageClassifier

# gaia lib
from gaiavision.core import DynamicMixin


@CLASSIFIERS.register_module()
class DynamicImageClassifier(ImageClassifier, DynamicMixin):
    # search_space = {'backbone', 'neck', 'roi_head', 'rpn_head'}
    search_space = {'backbone'}

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None):
        super(DynamicImageClassifier, self).__init__(
                backbone,
                neck=neck,
                head=head,
                pretrained=pretrained,
                train_cfg=train_cfg)

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)



