"""Base segmentation dataset"""
import os
import random
import numpy as np
import torchvision
import albumentations as A

from ...config import cfg

__all__ = ['SegmentationDataset']

class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        if root is None:
            self.root = cfg.DATASET.ROOT_PATH #os.path.join(cfg.DATASET.ROOT_PATH, root)
        else:
            self.root = root
            
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.crop_size = self.to_tuple(crop_size)

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):  # https://blog.csdn.net/zenghaitao0128/article/details/78509297
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def get_TrainAugmentation(self, p=1):
        train_transform = [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            # A.ShiftScaleRotate(scale_limit=(0.5, 1.5), rotate_limit=(-90, 90), shift_limit=0.1, p=1, border_mode=0),
            #A.RandomScale(scale_limit=0.3, p=0.3),
            #A.PadIfNeeded(512, 512, p=1),   #512, 512
            A.RandomRotate90(p=0.3),
            # # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            A.RandomCrop(height=cfg.TRAIN.CROP_SIZE, width=cfg.TRAIN.CROP_SIZE, always_apply=True),   #512
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.4, rotate_limit=90, p=0.3),
            A.GaussNoise(p=0.2),
            # A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # A.Sharpen(alpha=(0.1, 0.3), lightness=(0.3, 0.7), p=1),  # lightness=(0.5, 1.0)   #alpha=(0.2, 0.5)
            A.RandomBrightnessContrast(p=0.2),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=5, val_shift_limit=5, p=0.2),
            # A.OpticalDistortion(distort_limit=0.25, shift_limit=0.2, p=0.2),
            # A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.2),
            #A.RandomGamma(gamma_limit=46, p=0.5)
            # A.IAAPerspective(p=0.5),

            # A.OneOf([
            #         A.IAAAdditiveGaussianNoise(),
            #         A.GaussNoise(),
            #         A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # ], p=0.2),

            # A.OneOf(
            #     [
            #         # A.CLAHE(p=0.1),    #clahe supports only uint8 inputs
            #         A.RandomBrightnessContrast(p=0.5),
            #         #A.RandomGamma(p=0.5)
            #     ],
            #     p=0.9
            # ),
            #
            # A.OneOf(
            #     [
            #         A.HueSaturationValue(p=0.5),
            #         A.RGBShift(p=0.5),
            #     ],
            #     p=0.8
            # ),

        ]

        train_transform_2 = A.Compose([

            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            # A.ShiftScaleRotate(scale_limit=(0.5, 1.5), rotate_limit=(-90, 90), shift_limit=0.1, p=1, border_mode=0),
            # A.RandomScale(scale_limit=0.3, p=0.3),
            # A.PadIfNeeded(512, 512, p=1),   #512, 512
            A.RandomRotate90(p=0.3),
            # # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            A.RandomCrop(height=cfg.TRAIN.CROP_SIZE, width=cfg.TRAIN.CROP_SIZE, always_apply=True),  # 512
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.4, rotate_limit=90, p=0.3),
            A.OpticalDistortion(distort_limit=0.25, shift_limit=0.2, p=0.2),
            A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.2),
            # color transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.2),
                    A.RandomGamma(p=0.2),
                    A.ChannelShuffle(p=0.2),
                    # A.HueSaturationValue(p=1),
                    # A.RGBShift(p=1),
                ],
                p=0.5,
            ),

            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=0.2),
                    A.MultiplicativeNoise(p=0.2),
                    # A.ImageCompression(quality_lower=0.7, p=1),
                    A.GaussianBlur(p=0.2),
                ],
                p=0.2,
            ),
        ])

        return A.Compose(locals()[cfg.TRAIN.DATA_AUGMENT], p=p)  #A.Compose(train_transform, p=p)

    def get_StrongAugmentation(self, p=1):
        strong_transform = [
            A.VerticalFlip(p=0.3),
            A.RandomSizedCrop(p=0.3),
            A.RandomBrightness(),
            A.RandomContrast(),
            # A.Rotate(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.4, rotate_limit=90, p=0.3),
            A.OpticalDistortion(distort_limit=0.25, shift_limit=0.2, p=0.2),
            A.GaussianBlur(p=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=10, p=1),
            A.HueSaturationValue(p=0.2),
        ]

        return A.Compose(strong_transform)

    def get_ValAugmentation(self, p=1):
        val_transform = [
            A.PadIfNeeded(self.crop_size[0], self.crop_size[1])
        ]

        return A.Compose(val_transform)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            A.Lambda(image=preprocessing_fn),
            A.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return A.Compose(_transform)
