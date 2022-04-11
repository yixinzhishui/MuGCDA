"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset


class CitySegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/cityscapes'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='Cityscapes', split='train', mode=None, transform=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/cityscapes"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)      #np.unique：https://blog.csdn.net/yangyuwen_yang/article/details/79193770
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)   #np.digitize:https://blog.csdn.net/weixin_37532614/article/details/105724327?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242   https://blog.csdn.net/weixin_38358654/article/details/78997769
        return self._key[index].reshape(mask.shape)  #相当于将mask中0的位置变为7,1的位置变为8，即将  self._mapping  对应到self.valid_classes             #ravel():https://blog.csdn.net/yunfeather/article/details/106316811

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        # for root, _, files in os.walk(img_folder):   #https://www.runoob.com/python/os-walk.html   root的子目录也会遍历
        #     for filename in files:
        #         if filename.startswith('._'):
        #             continue
        #         if filename.endswith('.png'):
        #             imgpath = os.path.join(root, filename)
        #             foldername = os.path.basename(os.path.dirname(imgpath))    #os.path.dirname:https://blog.csdn.net/weixin_38470851/article/details/80367143
        #             maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
        #             maskpath = os.path.join(mask_folder, foldername, maskname)
        #             if os.path.isfile(imgpath) and os.path.isfile(maskpath):
        #                 img_paths.append(imgpath)
        #                 mask_paths.append(maskpath)
        #             else:
        #                 logging.info('cannot find the mask or image:', imgpath, maskpath)

        for root, dirs, _ in os.walk(img_folder):
            for dir_name in dirs:
                for filename in os.listdir(os.path.join(root, dir_name)):
                    if filename.endswith('.png'):
                        imgpath = os.path.join(img_folder, dir_name, filename)
                        maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                        maskpath = os.path.join(mask_folder, dir_name, maskname)
                        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                        else:
                            logging.info('cannot find the mask or img:', imgpath, maskpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit', split)
        mask_folder = os.path.join(folder, 'gtFine', split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        logging.info('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = CitySegmentation()
