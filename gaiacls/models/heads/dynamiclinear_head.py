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


    def loss(self, cls_score, gt_label, teacher_logits=None, only_distill=False, distillation_weight=0.4,**kwargs):

        #import pdb
        #pdb.set_trace()
        losses = dict()


        # 在有AugMix的时候，相当于对于超网是正常的AugMix训练，对于子网，所有图像都是按照蒸馏进行训练
        if only_distill:
            T = 2
            teacher_score = F.softmax(teacher_logits/T, dim=1)
            student_score = F.softmax(cls_score/T, dim=1)
            # 这个是不是需要裁剪一下，KLDivergence在两个分布差别很大的时候梯度好像是有问题的。
            temp_loss = torch.nn.KLDivLoss()(student_score.log(), teacher_score)
            losses['loss'] = temp_loss
            return losses

        num_samples = len(cls_score)
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        #print("1111")
            
        if teacher_logits is not None:
            T = 2
            teacher_score = F.softmax(teacher_logits/T, dim=1)
            student_score = F.softmax(cls_score/T, dim=1)
            # 这个是不是需要裁剪一下，KLDivergence在两个分布差别很大的时候梯度好像是有问题的。
            temp_loss = torch.nn.KLDivLoss()(student_score.log(),teacher_score)
            # 这个地方的蒸馏loss感觉最好也设置一个weight,先设置成0.4 具体设置多少最优感觉还得调参？
            losses['loss'] += (temp_loss*distillation_weight)  
            #print("2222")

        return losses




    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        cls_score = self.fc(img)
        
        if kwargs.get('return_logits',False):
            return cls_score
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.fc(x)
        
        losses = self.loss(cls_score, gt_label, teacher_logits=teacher_logits,**kwargs)    
        return_logits = kwargs.get('return_logits',False)
        if return_logits:
            losses['logits'] = cls_score.detach()
        return losses
