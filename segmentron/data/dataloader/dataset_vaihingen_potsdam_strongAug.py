import os
import numpy as np
from PIL import Image
import logging
import pickle
import pandas

import cv2
from osgeo import gdal

from torchvision import transforms
import torch.utils.data as data

#from .seg_data_base import SegmentationDataset
#from .dataset_base import SegmentationDataset
from segmentron.data.dataloader.dataset_base import SegmentationDataset
from segmentron.config import cfg



class StrongVaihingenPotsdamDataset(SegmentationDataset):

    #NUM_CLASS = cfg.DATASET.NUM_CLASSES #10
    def __init__(self, root=None, data_list_root=None, split='train', mode=None, transform=None, **kwargs):
        super(StrongVaihingenPotsdamDataset, self).__init__(root, split, mode, transform, **kwargs)

        assert os.path.exists(self.root), "the root of dataset do not exist:{}".format(self.root)
        if data_list_root is None:
            self.image_paths, self.mask_paths = self.get_pathList(self.root, self.split)
        else:
            self.image_paths, self.mask_paths = self.get_pathList_v4(self.root, data_list_root, self.split)


        self.classes_index = cfg.DATASET.CLASS_INDEX[0] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ignore_value = cfg.DATASET.IGNORE_INDEX #255

    # def encode_segmap(self, mask):
    #     if len(mask.shape) == 2:
    #         encode_mask = np.zeros(mask.shape, dtype=np.uint8)
    #         for ii, class_index in enumerate(self.classes_index):
    #             encode_mask[mask == class_index] = ii
    #         encode_mask[mask == self.ignore_value] = 0
    #     else:
    #         encode_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    #         for ii, class_index in enumerate(self.classes_index):
    #             encode_mask[np.all(mask == class_index, 2)] = ii
    #         encode_mask[np.all(mask == [255, 255, 255], 2)] = 255
    #         encode_mask = np.array(encode_mask)
    #     return encode_mask

    def encode_segmap(self, mask):
        if len(mask.shape) == 2:
            encode_mask = np.zeros(mask.shape, dtype=np.uint8)
            for class_index, pixel_value in self.classes_index.items():
                encode_mask[mask == pixel_value] = class_index
            encode_mask[mask == 0] = 255
        else:
            encode_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for class_index, pixel_value in self.classes_index.items():
                encode_mask[np.all(mask == pixel_value, 2)] = class_index
            encode_mask = np.array(encode_mask)
        return encode_mask

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    # def readImage(self, dirname):
    #
    #     image = cv2.imread(dirname, -1)
    #
    #     return image

    def __getitem__(self, item):

        img = self.readImage(self.image_paths[item])
        if self.mode == 'test':
            if self.transform is not None:
                # img = self.ImageNormalize(img)
                img = self.transform(img)
            return img, os.path.basename(self.image_paths[item])

        mask = self.readImage(self.mask_paths[item])

        if self.mode == 'train':
            sample = self.get_TrainAugmentation(0.7)(image=img.astype(np.uint8), mask=mask.astype(np.uint8))
            img, mask = sample['image'], sample['mask']
            sample_strong = self.get_StrongAugmentation(p=1)(image=img, mask=mask)

            mask = self.encode_segmap(mask)
        elif self.mode == 'val':
            sample = self.get_ValAugmentation()(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

            mask = self.encode_segmap(mask)
        else:
            assert self.mode == 'testval'
            mask = self.encode_segmap(mask)

        if self.transform is not None:
            # img = self.ImageNormalize(img)
            img = self.transform(img)
        return img, mask, self.image_paths[item]

    def __len__(self):
        return len(self.image_paths)

    # def get_pathList_v2(self, folder, dirname, split='train'):
    #     img_paths = []
    #     mask_paths = []
    #     with open(dirname, "r") as handle:
    #         content = handle.readlines()
    #     for img_name in content:
    #         img_name = img_name.strip()
    #         imgpath = os.path.join(folder,  split, "images", img_name)
    #         maskpath = os.path.join(folder,  split, "labels", img_name)
    #         if os.path.exists(imgpath) or os.path.exists(maskpath):
    #             img_paths.append(imgpath)
    #             mask_paths.append(maskpath)
    #
    #     return img_paths, mask_paths

    def get_pathList_v2(self, folder, dirname, split='train'):
        img_paths = []
        mask_paths = []
        with open(dirname, "r") as handle:
            content = handle.readlines()
        for img_name in content:
            img_name = img_name.strip()
            imgpath = os.path.join(folder, "images", img_name.replace('.png', '.tif'))
            maskpath = os.path.join(folder, "labels", img_name)
            if os.path.exists(imgpath) or os.path.exists(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), folder))
        return img_paths, mask_paths

    def get_pathList_v3(self, folder, dirname, split='train'):
        img_paths = []
        mask_paths = []
        self.file_to_index = dict()
        self.file_to_label, self.label_to_file, _ = pickle.load(open(dirname, "rb"))
        for index, img_name in enumerate(self.file_to_label.keys()):
            imgpath = os.path.join(folder, split, "images", img_name.replace('.png', '.tif'))  #img_name
            maskpath = os.path.join(folder, split, "labels", img_name)
            if os.path.exists(imgpath) or os.path.exists(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)

            self.file_to_index[img_name] = index

        logging.info('Found {} images in the folder {}'.format(len(img_paths), folder))
        return img_paths, mask_paths
        # img_list = []
        # class_filecount= dict()
        # for i in range(cfg.DATASET.NUM_CLASSES):
        #     class_filecount[i] = 0
        # cur_class_dist = np.zeros(cfg.DATASET.NUM_CLASSES)
        # for item in len(file_to_label):
        #     if cur_class_dist.sum() == 0:
        #         dist = cur_class_dist.copy()
        #     else:
        #         dist = cur_class_dist / cur_class_dist.sum()
        #
        #     w = 1 / np.log(1 + 1e2 + dist)
        #     w = w / w.sum()
        #     c = np.random.choice(cfg.DATASET.NUM_CLASSES, p=w)
        #
        #     if class_filecount[c] > (len(label_to_file[c]) - 1):
        #         np.random.shuffle(label_to_file[c])
        #         class_filecount[c] = class_filecount[c] % (len(label_to_file[c]) - 1)


    def get_pathList_v4(self, folder, dirname, split='train'):
        img_paths = []
        mask_paths = []

        if split == 'train':
            df = pandas.read_csv(dirname)
            for row in range(len(df)):
                if int(df['fold'][row]) == 0:
                    dir_path = os.path.join(folder, df['name'][row])
                    img_list = os.listdir(dir_path)
                    for filename in img_list:
                        if 'image' in filename:
                            img_path = os.path.join(dir_path, filename)
                            mask_path = os.path.join(dir_path, filename)
                            if os.path.exists(img_path) and os.path.exists(mask_path):
                                img_paths.append(img_path)
                                mask_paths.append(mask_path)
        elif split == 'val':
            df = pandas.read_csv(dirname)
            for row in range(len(df)):
                if int(df['fold'][row]) == 0:
                    dir_path = os.path.join(folder, df['name'][row])
                    img_list = os.listdir(dir_path)
                    for filename in img_list:
                        if 'image' in filename:
                            img_path = os.path.join(dir_path, filename)
                            mask_path = os.path.join(dir_path, filename)
                            if os.path.exists(img_path) and os.path.exists(mask_path):
                                img_paths.append(img_path)
                                mask_paths.append(mask_path)
        elif split == 'test':
            img_dir = os.path.join(folder, 'test')
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    img_paths.append(img_path)
        logging.info('Found {} images in the {} folder {}'.format(len(img_paths), split, folder))
        return img_paths, mask_paths

    def get_pathList(self, folder, split='train'):
        def get_pathPairs(img_folder, mask_folder):
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
                    assert len(img_paths) == len(mask_paths), "images not equal masks"
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
                    assert len(img_paths) == len(mask_paths), "images not equal masks"
                    return img_paths, mask_paths

        def get_pathImage(img_folder):
            img_paths = []

            for root, dirs, filenames in os.walk(img_folder, topdown=True):
                if len(dirs) != 0:
                    for dir in dirs:
                        for filename in os.listdir(os.path.join(root, dir)):
                            if filename.endswith('.png') or filename.endswith('.tif'):
                                imgpath = os.path.join(img_folder, dir, filename)

                                if os.path.isfile(imgpath):
                                    img_paths.append(imgpath)
                                else:
                                    logging.info('cannot find the img', imgpath)
                    logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
                    return img_paths
                else:
                    for filename in filenames:
                        imgpath = os.path.join(root, filename)
                        # maskname = filename.replace('', '')

                        if os.path.isfile(imgpath):
                            img_paths.append(imgpath)
                        else:
                            logging.info('cannot find the img or mask', imgpath)

                    logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
                    return img_paths

        if split in ['train', 'val', 'pesudo_train']:
            img_folder = os.path.join(folder, 'images')  #split, 'images'
            mask_folder = os.path.join(folder, 'labels')  #split, 'labels'
            img_paths, mask_paths = get_pathPairs(img_folder, mask_folder)
            return img_paths, mask_paths
        elif split in ['trainval']:
            logging.info('trainval set')

            train_img_folder = os.path.join(folder, 'train', 'images')
            train_mask_folder = os.path.join(folder, 'train', 'labels')
            val_img_folder = os.path.join(folder, 'val', 'images')
            val_mask_folder = os.path.join(folder, 'val', 'labels')

            train_img_paths, train_mask_paths = get_pathPairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_pathPairs(val_img_folder, val_mask_folder)

            img_paths = train_img_paths + val_img_paths
            mask_paths = val_img_paths + val_mask_paths
            return img_paths, mask_paths
        else:
            assert split == "test", "dataset split is {}, but expect 【test】".format(split)
            test_img_folder = os.path.join(folder, split)  #folder

            test_img_paths = get_pathImage(test_img_folder)
            return test_img_paths, []

    def ImageNormalize(self, image, max=None, min=None):
        if max is None:
            max = 255.0
        if min is None:
            min = 0
        return ((image - min) / (max - min)).astype(np.float32)

if __name__ == '__main__':
    import cv2
    import os

    def Write_Image(filename, img_data, img_proj=None, img_geotrans=None):

        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        if img_proj != None:
            image.SetProjection(img_proj)
        if img_geotrans != None:
            image.SetGeoTransform(img_geotrans)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i+1).WriteArray(img_data[i])

        del image

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3539229, 0.3734769, 0.34095916], [0.23289922, 0.22774169, 0.22549641]),
    ])

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'crop_size': cfg.TRAIN.CROP_SIZE}
    dataset = GDALLandcoverCommon_Segmentation(root=r'J:\Open_Datasets\LULC\GID\GID_15_512x512', split='train', mode='train', **data_kwargs)
    train_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    for ii, [images, targets, _] in enumerate(train_dataloader):
        cv2.imwrite(os.path.join(r"D:\Miscellaneous\picture\temp\sample_build_crop", _[0]), targets[0].numpy())
        image = images[0].numpy()
        Write_Image(os.path.join(r"D:\Miscellaneous\picture\temp", _[0]), image)
        break





