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

# to do change from mmcls（）
#  这个从pytorch_imgage_models里面改的有点问题，它是在image已经是（0,1）的基础上进行的
#  跟mmcls的接口不统一，这个在mmlab里面是按照放在normalize之前的，所以mmlab里面重写了一下RandomErasing。
@PIPELINES.register_module()
class RandomErasing_augmix(object):
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
                        dtype=dtype)
                    break

    def __call__(self, results):
        
        for key in results.get('img_fields', ['img']):
            img = results[key]
            assert img.shape[-1] == self.num_splits*self.origin_img_channel

            if(self.num_splits == 1):
                input = img
                self._erase(input, *input.shape, input.dtype)
            else:
                # Only add random erasing in extra image, if wanna add it in all splits, just
                # make this transform before AugMix in the configfile.
                for i in range(1,self.num_splits):
                    input = img[:,:, i*self.origin_img_channel:(i+1)*self.origin_img_channel] # 赋值是浅拷贝
                    self._erase(input, *input.shape, input.dtype)
                    
        return results


@PIPELINES.register_module()
class Normalize_augmix(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True, num_splits=1, origin_img_channel=3):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.num_splits = num_splits
        self.origin_img_channel = origin_img_channel

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            
            assert results[key].shape[-1] == self.num_splits*self.origin_img_channel
            img_list = []
            for i in range(self.num_splits):
                start = i*self.origin_img_channel
                end = start + self.origin_img_channel
                
                temp_result = mmcv.imnormalize(results[key][:,:,start:end], self.mean, self.std,
                                            self.to_rgb)
                img_list.append(temp_result)
        results[key] = np.concatenate(img_list,axis=-1)
        
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str








