""" Transforms for GeoNRW and DFC2020 samples.
Samples are dictionaries, with the keys as the datatype (seg, rgb, dem or sar)
"""

import random
import numbers
import torch
import PIL.Image
import torchvision.transforms.functional as TF
import torch.nn.functional as FN

# Segmentation maps require nearest neighbour resampling to preserve discrete classes.
# Available resample methods
# https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-filters.
RESAMPLE_METHODS = {
    "seg": PIL.Image.NEAREST,
    "rgb": PIL.Image.BILINEAR,
    "dem": PIL.Image.BILINEAR,
    "sar": PIL.Image.BILINEAR,
}


class ToTensor:
    def __call__(self, sample):
        return {k: TF.to_tensor(v) for k, v in sample.items()}

class OneHot:

    def __call__(self, sample):
        for k, v in sample.items():
            if k == 'seg':
                sample[k] = FN.one_hot(v.long(), 3).squeeze().permute(2, 0, 1).float()
        return sample

class SmoothDem:

    def __call__(self, sample):
        for k, v in sample.items():
            if k == 'dem':
                small = FN.interpolate(v.unsqueeze(0), (12,12), mode='nearest')
                sample[k] = FN.interpolate(small, (192, 192), mode='bilinear').squeeze(0)
        return sample

class TensorApply:
    """ Applies functions to some datatype of a sample """

    valid_kwargs = set(RESAMPLE_METHODS.keys())

    def __init__(self, **kwargs):
        """ Pass functions as keyword arguments.
        The key defines the datatype the function operatoes on

        """
        if not self.valid_kwargs.issuperset(set(kwargs.keys())):
            raise ValueError(
                "Keywords must be chosen from {}".format(self.valid_kwargs)
            )
        self.funcs = kwargs

    def __call__(self, sample):
        for key, func in self.funcs.items():
            sample[key] = func(sample[key])

        return sample


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        w, h = 200, 200
        try:
            i = random.randrange(0, h - self.size[1])
        except ValueError:  # empty range because image is too small for crop
            i = 0
        try:
            j = random.randrange(0, w - self.size[0])
        except ValueError:  # empty range
            j = 0

        return {k: TF.crop(v, i, j, *self.size) for k, v in sample.items()}


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {k: TF.hflip(v) for k, v in sample.items()}
        return sample

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {k: TF.vflip(v) for k, v in sample.items()}
        return sample


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        return {k: TF.center_crop(v, self.size) for k, v in sample.items()}


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        # resize crop to desired dimensions
        return {
            k: TF.resize(v, self.size, RESAMPLE_METHODS[k]) for k, v in sample.items()
        }
