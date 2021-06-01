# refer from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/jsd.py
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

@LOSSES.register_module()
class JsdCrossEntropyLoss(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.1, reduction='mean', loss_weight=1.0):
        super(JsdCrossEntropyLoss,self).__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, cls_score, label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # pdb.set_trace()
        split_size = cls_score.shape[0] // self.num_splits
        assert split_size * self.num_splits == cls_score.shape[0]
        logits_split = torch.split(cls_score, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], label)
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        
        loss_cls = loss * self.loss_weight
        return loss_cls
