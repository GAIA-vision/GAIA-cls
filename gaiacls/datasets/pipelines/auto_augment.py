import copy
import random
from numbers import Number

import mmcv
import numpy as np

from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines.compose import Compose

@PIPELINES.register_module()
class RandAugment(object):
    """Random augmentation. This data augmentation is proposed in `RandAugment:
    Practical automated data augmentation with a reduced search space.
    <https://arxiv.org/abs/1909.13719>`_.
    Args:
        policies (list[dict]): The policies of random augmentation. Each
            policy in ``policies`` is one specific augmentation policy (dict).
            The policy shall at least have key `type`, indicating the type of
            augmentation. For those which have magnitude, (given to the fact
            they are named differently in different augmentation, )
            `magnitude_key` and `magnitude_range` shall be the magnitude
            argument (str) and the range of magnitude (tuple in the format of
            (val1, val2)), respectively. Note that val1 is not necessarily
            less than val2.
        num_policies (int): Number of policies to select from policies each
            time.
        magnitude_level (int | float): Magnitude level for all the augmentation
            selected.
        total_level (int | float): Total level for the magnitude. Defaults to
            30.
        magnitude_std (Number | str): Deviation of magnitude noise applied.
            If positive number, magnitude is sampled from normal distribution
                (mean=magnitude, std=magnitude_std).
            If 0 or negative number, magnitude remains unchanged.
            If str "inf", magnitude is sampled from uniform distribution
                (range=[min, magnitude]).
    Note:
        `magnitude_std` will introduce some randomness to policy, modified by
        https://github.com/rwightman/pytorch-image-models
        When magnitude_std=0, we calculate the magnitude as follows:
        .. math::
            magnitude = magnitude_level / total_level * (val2 - val1) + val1
    """

    def __init__(self,
                 policies,
                 num_policies,
                 magnitude_level,
                 magnitude_std=0.,
                 total_level=30,
                 num_splits=1, 
                 origin_img_channel=3):
        assert isinstance(num_policies, int), 'Number of policies must be ' \
            f'of int type, got {type(num_policies)} instead.'
        assert isinstance(magnitude_level, (int, float)), \
            'Magnitude level must be of int or float type, ' \
            f'got {type(magnitude_level)} instead.'
        assert isinstance(total_level, (int, float)),  'Total level must be ' \
            f'of int or float type, got {type(total_level)} instead.'
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, dict) and 'type' in policy, \
                'Each policy must be a dict with key "type".'

        assert isinstance(magnitude_std, (Number, str)), \
            'Magnitude std must be of number or str type, ' \
            f'got {type(magnitude_std)} instead.'
        if isinstance(magnitude_std, str):
            assert magnitude_std == 'inf', \
                'Magnitude std must be of number or "inf", ' \
                f'got "{magnitude_std}" instead.'

        assert num_policies > 0, 'num_policies must be greater than 0.'
        assert magnitude_level >= 0, 'magnitude_level must be no less than 0.'
        assert total_level > 0, 'total_level must be greater than 0.'

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.policies = self._process_policies(policies)
        self.num_splits = num_splits
        self.origin_img_channel = origin_img_channel

    def _process_policies(self, policies):
        processed_policies = []
        for policy in policies:
            processed_policy = copy.deepcopy(policy)
            magnitude_key = processed_policy.pop('magnitude_key', None)
            if magnitude_key is not None:
                val1, val2 = processed_policy.pop('magnitude_range')
                magnitude_value = (self.magnitude_level / self.total_level
                                   ) * float(val2 - val1) + val1

                # if magnitude_std is positive number or 'inf', move
                # magnitude_value randomly.
                maxval = max(val1, val2)
                minval = min(val1, val2)
                if self.magnitude_std == 'inf':
                    magnitude_value = random.uniform(minval, magnitude_value)
                elif self.magnitude_std > 0:
                    magnitude_value = random.gauss(magnitude_value,
                                                   self.magnitude_std)
                    magnitude_value = min(maxval, max(minval, magnitude_value))
                processed_policy.update({magnitude_key: magnitude_value})
            processed_policies.append(processed_policy)
        return processed_policies

    def __call__(self, results):
        if self.num_policies == 0:
            return results
        sub_policy = random.choices(self.policies, k=self.num_policies)
        sub_policy = Compose(sub_policy)

        for key in results.get('img_fields', ['img']):
            img = results[key]
            assert img.shape[-1] == self.num_splits*self.origin_img_channel

            if(self.num_splits == 1):
                return sub_policy(results)
            else:
                # Only add RandAugment in extra image, if wanna add it in all splits, just
                # make this transform before AugMix in the configfile.
                record_img = copy.deepcopy(img)
                for i in range(1,self.num_splits):
                    results[key] = img[:,:, i*self.origin_img_channel:(i+1)*self.origin_img_channel]
                    results = sub_policy(results)
                    record_img[:,:, i*self.origin_img_channel:(i+1)*self.origin_img_channel] = results[key]
                results[key] = record_img
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies}, '
        repr_str += f'num_policies={self.num_policies}, '
        repr_str += f'magnitude_level={self.magnitude_level}, '
        repr_str += f'total_level={self.total_level})'
        return repr_str