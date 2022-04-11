import os
import numpy as np
from PIL import Image
import logging
import cv2

from torchvision import transforms
import torch.utils.data as data

#from .seg_data_base import SegmentationDataset
from segmentron.data.dataloader.seg_data_base import SegmentationDataset
from segmentron.config import cfg



class Landcover9num_Segmentation(SegmentationDataset):

    NUM_CLASS = cfg.DATASET.NUM_CLASSES #10
    def __init__(self, root='', split='train', mode=None, transform=None, **kwargs):
        super(Landcover9num_Segmentation, self).__init__(root, split, mode, transform, **kwargs)

        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/cityscapes"
        self.image_paths, self.mask_paths = get_FADA_pairs(self.root, self.split)
        assert len(self.image_paths) == len(self.mask_paths), "images not equal masks"

        self.classes_index = cfg.DATASET.CLASS_INDEX #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ignore_value = cfg.DATASET.IGNORE_INDEX #255

    def encode_segmap(self, mask):
        if len(mask.shape) == 2:
            encode_mask = np.zeros(mask.shape, dtype=np.uint8)
            for ii, class_index in enumerate(self.classes_index):
                encode_mask[mask == class_index] = ii
            encode_mask[mask == self.ignore_value] = 0
        else:
            encode_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for ii, class_index in enumerate(self.classes_index):
                encode_mask[np.all(mask == class_index, 2)] = ii
            encode_mask = np.array(encode_mask).astype('int32')
        return encode_mask

    def __getitem__(self, item):
        #img = Image.open(self.image_paths[item])
        img = Image.fromarray(cv2.imread(self.image_paths[item], -1))
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.image_paths[item])
        #mask = Image.open(self.mask_paths[item])
        mask = Image.fromarray(cv2.imread(self.mask_paths[item], -1))
        #mask.save(os.path.join(r"D:\Miscellaneous\picture\temp",os.path.basename(self.mask_paths[item])))  #debug_clause
        #print(mask.size)    #debug_clause
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)

            mask = self.encode_segmap(mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
            mask = self.encode_segmap(mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
            mask = self.encode_segmap(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.image_paths[item])

    def __len__(self):
        return len(self.image_paths)


def get_FADA_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []

        for root, dirs, filenames in os.walk(img_folder, topdown=True):
            if len(dirs) != 0:
                for dir in dirs:
                    for filename in os.listdir(os.path.join(root, dir)):
                        if filename.endswith('.png') or filename.endswith('.tif'):
                            imgpath = os.path.join(img_folder, dir, filename)
                            #maskname = filename.replace('', '')
                            maskpath = os.path.join(mask_folder, dir, filename)

                            if os.path.isfile(imgpath) or os.path.isfile(maskpath):
                                img_paths.append(imgpath)
                                mask_paths.append(maskpath)
                            else:
                                logging.info('cannot find the img or mask', imgpath, maskpath)
                logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
                return img_paths, mask_paths
            else:
                for filename in filenames:
                    imgpath = os.path.join(root, filename)
                    # maskname = filename.replace('', '')
                    maskpath = os.path.join(mask_folder, filename)   #.replace('.tif', '.png')

                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logging.info('cannot find the img or mask', imgpath, maskpath)

                logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
                return img_paths, mask_paths


    if split in ['train', 'val']:          #训练集划分：https://www.cnblogs.com/marsggbo/p/10496696.html   https://blog.csdn.net/l8947943/article/details/105623150
        img_folder = os.path.join(folder, split, 'images')  #split, 'images'   'images', split
        mask_folder = os.path.join(folder, split, 'labels')  #split, 'labels'   'labels', split
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        logging.info('trainval set')

        train_img_folder = os.path.join(folder, 'train', 'images')
        train_mask_folder = os.path.join(folder, 'train', 'labels')
        val_img_folder = os.path.join(folder, 'val', 'images')
        val_mask_folder = os.path.join(folder, 'val', 'labels')

        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)

        img_paths = train_img_paths + val_img_paths
        mask_paths = val_img_paths + val_mask_paths
        return img_paths, mask_paths

if __name__ == '__main__':
    import cv2
    import os

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                   'crop_size': cfg.TRAIN.CROP_SIZE}
    dataset = Landcover9num_Segmentation(root=r'J:\Expert_Datasets\debug', split='train', mode='train', **data_kwargs)
    train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    for ii, [images, targets, _] in enumerate(train_dataloader):
        cv2.imwrite(os.path.join(r"D:\Miscellaneous\picture\temp", _[0]), targets[0].numpy())
        pass




