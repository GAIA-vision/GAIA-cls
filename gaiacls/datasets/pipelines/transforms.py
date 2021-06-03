import inspect
import math
import random
import pdb

import mmcv
import numpy as np

from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines.compose import Compose

import copy

try:
    import albumentations
except ImportError:
    albumentations = None


def _get_pixels(per_pixel, rand_color, patch_size, dtype=np.float32):
    '''
    refer from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
    '''
    if per_pixel:
        return np.random.normal(0,1,patch_size).astype(dtype)
    elif rand_color:
        return np.random.normal(0,1,(patch_size[0], 1, 1)).astype(dtype)
    else:
        return np.zeros((patch_size[0], 1, 1), dtype=dtype)


@PIPELINES.register_module()
class AugMix(object):
    """
    may be RepeatImg better ?
    """

    def __init__(self, aug_splits=3):
        self.aug_splits=aug_splits

    def __call__(self, results):
        # 在ImageToTensor之前都是以ndarray形式处理，并且是[H,W,C]的格式
        for key in results.get('img_fields', ['img']):
            img = results[key]
            assert img.ndim == 3
            H,W,C = img.shape

            temp_img = np.zeros((H, W, C*self.aug_splits,),dtype=img.dtype)
            for i in range(self.aug_splits):
                temp_img[:,:,i*C:(i+1)*C] = img           
            results[key] = temp_img
            
        return results

    # ToDo
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

# 因为本身mmcls的AutoAugment已经注册过了，所以不能再用这个名字
# 这里加个后缀_augmix
# 还得check下mmcls的AutoAugment和pytorch image models的那个AutoAugment是否可以互相替代
@PIPELINES.register_module()
class AutoAugment_augmix(object):
    """
    """
    def __init__(self,):
        pass

    def __call__(self,):
        
        pass
        return results

    def __repr__(self):
        
        repr_str = self.__class__.__name__
        pass
    
        return repr_str


@PIPELINES.register_module()
class RandomErasing(object):
    """
        refer from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
        Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=1, origin_img_channel=3):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.origin_img_channel = origin_img_channel
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, img_h, img_w, chan, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[top:top + h, left:left + w, :] = _get_pixels(
                        self.per_pixel, self.rand_color, (h, w, chan),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, results):

        for key in results.get('img_fields', ['img']):
            img = results[key]
            assert img.shape[0] == num_splits*origin_img_channel
            
            for i in range(1,self.num_splits):
                input = img[:,:, i*self.origin_img_channel:(i+1)*self.origin_img_channel] # 赋值是浅拷贝
                self._erase(input, *input.size(), input.dtype)
                    
        return results











