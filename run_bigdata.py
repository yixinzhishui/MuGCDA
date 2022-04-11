# 分块读取
import os
import sys
import torch
import cv2
import time

from osgeo import gdal
import numpy as np
import time
from tqdm import tqdm

from torchvision import transforms
import torch.nn as nn
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
import segmentron.solver.ttach as ttach

import torch.utils.data as data

from segmentron.config import cfg

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

Patch_Size = 512
Stride = 300


def set_batch_norm_attr(named_modules, attr, value):
    for m in named_modules:
        if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
            setattr(m[1], attr, value)

class SegmentationInterpret_BigData():
    def __init__(self, filename, patch_size=512, stride=300, transform=None, rw_mode="cv", **kwargs):
        self.filename = filename
        self.patch_size = patch_size
        self.stride = stride
        self.transform=transform
        self.rw_mode = rw_mode

        if rw_mode == "gdal":
            self.image = gdal.Open(filename)
            self.img_width = self.image.RasterXSize
            self.img_height = self.image.RasterYSize

            self.img_geotrans = self.image.GetGeoTransform()
            self.img_proj = self.image.GetProjection()
            self.get_RasterCount()
        else:
            self.img_data = cv2.imread(filename, -1)
            #self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2RGB)
            #self.img_data = np.rollaxis(self.img_data, 2, 0)
            self.img_height, self.img_width, self.img_bands = self.img_data.shape

            self.pred_big_data = np.zeros((self.img_height, self.img_width), dtype=np.float16)

            self.img_index_list = self.get_PatchIndexList(self.img_height, self.img_width)

    def get_PatchIndexList(self, img_height, img_width):

        img_index_list = []

        def get_ImageSlide_steps(image_height, image_width):
            height_steps = [h_steps if h_steps + self.patch_size < image_height else image_height - self.patch_size for h_steps in
                            range(0, image_height, self.stride)]
            width_steps = [w_steps if w_steps + self.patch_size < image_width else image_width - self.patch_size for w_steps in
                           range(0, image_width, self.stride)]
            height_steps = np.unique(height_steps)
            width_steps = np.unique(width_steps)
            return height_steps, width_steps

        def get_EdgeCrop_sizes(height_steps, width_steps):
            patch_h_overlap = (np.array(height_steps) + self.patch_size)[:-1] - np.array(height_steps)[1:]
            patch_w_overlap = (np.array(width_steps) + self.patch_size)[:-1] - np.array(width_steps)[1:]
            h_top, h_bottom = list(map(lambda x: int(x), [0] + list(patch_h_overlap / 2))), list(
                map(lambda x: int(x), list(-1 * patch_h_overlap / 2) + [0]))
            w_left, w_right = list(map(lambda x: int(x), [0] + list(patch_w_overlap / 2))), list(
                map(lambda x: int(x), list(-1 * patch_w_overlap / 2) + [0]))

            return [h_top, h_bottom], [w_left, w_right]

        height_steps, width_steps = get_ImageSlide_steps(img_height, img_width)
        h_EdgeCropSize, w_EdgeCropSize = get_EdgeCrop_sizes(height_steps, width_steps)

        for ii, h_step in enumerate(height_steps):
            for jj, w_step in enumerate(width_steps):
                img_index_list.append([h_step, w_step, [h_EdgeCropSize[0][ii], h_EdgeCropSize[1][ii]], [w_EdgeCropSize[0][jj], w_EdgeCropSize[1][jj]]])

        return img_index_list


    def get_RasterCount(self):
        if len(self.img_data.shape) == 2:
            self.img_bands = 1
        elif len(self.img_data.shape) == 3:
            self.img_bands = self.img_data.shape[0]

    def get_ImageData_Patch(self, w, h, w_size, h_size):

        return self.image.ReadAsArray(int(w), int(h), int(w_size), int(h_size))

    def set_ImageData(self):
        self.img_data = self.image.ReadAsArray(0, 0, self.img_width, self.img_height)

    def __len__(self):
        return len(self.img_index_list)

    def read_PatchImage(self, img_index):
        h_step, w_step, _, _ = img_index   # h_EdgeCropSize, w_EdgeCropSize
        patch_image = self.img_data[h_step:h_step + self.patch_size, w_step:w_step + self.patch_size, :]
        #patch_image = self.Get_ImageData_Patch(w_step, h_step, self.patch_size, self.patch_size)

        return patch_image


    def __getitem__(self, item):
        img_index = self.img_index_list[item]
        img = self.read_PatchImage(img_index)

        if self.transform is not None:
            #img = self.ImageNormalize(img)
            img = self.transform(img.astype(np.uint8))

        return img, img_index


    def ImageNormalize(self, image, max=None, min=None):
        if max is None:
            max = 255
        if min is None:
            min = 0
        return ((image - min) / (max - min)).astype(np.float32)   #网络权重为torch.cuda.FloatTensor类型，输入网络的数据类型需与网络权重类型一致

    def set_BigData(self, images, image_indexs):
        for item, image_index in enumerate(image_indexs):
            h_step, w_step, h_EdgeCropSize, w_EdgeCropSize = image_index
            image = image_indexs[item]
            #self.pred_big_data[h_step:h_step + Patch_Size, w_step:w_step + Patch_Size] = image
            self.pred_big_data[h_step + h_EdgeCropSize[0]:h_step + self.patch_size + h_EdgeCropSize[1], w_step + w_EdgeCropSize[0]:w_step + self.patch_size + w_EdgeCropSize[1]] = image[h_EdgeCropSize[0]:self.patch_size + h_EdgeCropSize[1], w_EdgeCropSize[0]:self.patch_size + w_EdgeCropSize[1]]

    def get_BigData(self):

        return self.pred_big_data

    def write_BigData(self, filename, image=None, mode="cv"):
        if image is None:
            save_image = self.pred_big_data
        else:
            save_image = image

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
                    image.GetRasterBand(i + 1).WriteArray(img_data[i])

            del image

        if os.path.isdir(filename):
            save_path = os.path.join(filename, os.path.join(self.filename))
        elif os.path.isfile(filename):
            save_path = filename
        if mode == "gdal":
            Write_Image(save_path, save_image, img_proj=self.img_proj, img_geotrans=self.img_geotrans)
        else:
            cv2.imwrite(save_path, save_image)


def get_ImageSlide_steps(image_height, image_width):
    height_steps = [h_steps if h_steps + Patch_Size < image_height else image_height - Patch_Size for h_steps in
                    range(0, image_height, Stride)]
    width_steps = [w_steps if w_steps + Patch_Size < image_width else image_width - Patch_Size for w_steps in
                   range(0, image_width, Stride)]
    height_steps = np.unique(height_steps)
    width_steps = np.unique(width_steps)
    return height_steps, width_steps


def get_EdgeCrop_sizes(height_steps, width_steps):
    patch_h_overlap = (np.array(height_steps) + Patch_Size)[:-1] - np.array(height_steps)[1:]
    patch_w_overlap = (np.array(width_steps) + Patch_Size)[:-1] - np.array(width_steps)[1:]
    h_top, h_bottom = list(map(lambda x: int(x), [0] + list(patch_h_overlap / 2))), list(
        map(lambda x: int(x), list(-1 * patch_h_overlap / 2) + [0]))
    w_left, w_right = list(map(lambda x: int(x), [0] + list(patch_w_overlap / 2))), list(
        map(lambda x: int(x), list(-1 * patch_w_overlap / 2) + [0]))

    return [h_top, h_bottom], [w_left, w_right]


def Get_datasetList(images_path):
    images_list = []

    for file in os.listdir(images_path):
        if file.endswith((".img")) or file.endswith((".tif")) or file.endswith((".png")):
            images_list.append(os.path.join(images_path, file))

    return images_list




class BigDataExtraction():
    def __init__(self, input_path, output_path):
        self.device = torch.device('cuda')
        self.input_path = input_path
        self.output_path = output_path

        self.input_pathList = self.get_DatasetList(input_path)

        # image transform
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        self.ttach_transform = ttach.aliases.d4_transform()

        self.model = get_segmentation_model().to(self.device)
        self.model, _, _, _, _ = load_model_resume(self.model)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            print('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        # if args.distributed:
        #     self.model = nn.parallel.DistributedDataParallel(self.model,
        #         device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)


    def get_DatasetList(self, images_path):
        images_list = []
        if os.path.isfile(images_path):
            images_list.append(images_path)
        elif os.path.isdir(images_path):
            for file in os.listdir(images_path):
                if file.endswith((".img")) or file.endswith((".tif")):
                    images_list.append(os.path.join(images_path, file))
        else:
            print("Input:{} is not the file or the path".format(images_path))

        return images_list

    def decode_segmap(self, mask):
        assert len(mask.shape) == 2, "the len of mask unexpect"
        assert cfg.DATASET.NUM_CLASSES == len(
            cfg.DATASET.CLASS_INDEX), "the value of NUM_CLASSES do not equal the len of CLASS_INDEX"
        height, width = mask.shape

        if isinstance(cfg.DATASET.CLASS_INDEX[0], int):
            decode_mask = np.zeros((height, width), dtype=np.uint)

            for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
                decode_mask[mask == index] = pixel
        else:
            decode_mask = np.zeros((height, width, 3), dtype=np.uint)
            for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
                decode_mask[mask == index] = pixel


    def extract_BigImage(self):

        self.model.eval()
        # if self.args.distributed:
        #     model = self.model.module
        # else:
        model = self.model
        time_start = time.time()
        for filename in self.input_pathList:
            img_dataset = SegmentationInterpret_BigData(filename, patch_size=512, stride=300, transform=self.input_transform)
            img_patch_loader = data.DataLoader(dataset=img_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.DATASET.WORKERS, pin_memory=True)

            for ii, (images, images_index) in tqdm(enumerate(img_patch_loader)):
                images = images.to(self.device)
                print(images.shape)
                with torch.no_grad():
                    preds_aug = []
                    # output = model.evaluate(image)
                    if cfg.TEST.USE_TTA:

                        # model = ttach.SegmentationTTAWrapper(model, ttach.aliases.hflip_transform(), merge_mode="mean")

                        for transform in self.ttach_transform:
                            augmented_image = transform.augment_image(images)
                            model_output =model(augmented_image)
                            deaug_output = transform.deaugment_mask(model_output)
                            preds_aug.append(deaug_output)

                    output = model(images)
                    preds_aug.append(output)
                    print(len(preds_aug))
                    output = torch.mean(torch.stack(preds_aug), 0)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    output = torch.argmax(output, 1)

                preds = output.cpu().data.numpy().astype(np.uint8)
                img_dataset.set_BigData(preds, images_index)

                pred_bigdata = img_dataset.get_BigData()
                pred_bigdata = self.decode_segmap(pred_bigdata)
                img_dataset.write_BigData(self.output_path, pred_bigdata)



if __name__ == '__main__':
    input_fileDir = "/input_path"  # r"/input_path" #sys.argv[1] #r"J:\Expert_Datasets\绿地识别\影像数据\500109北碚区\500109PL1+WV3+BJ2+GE1+GF2DOM01.img"  #E:\data\projects_and_topic\LULC_HuangHua\image_patch_GuiXi\360681BJ2+AP0DOM02_2_1.tif   #J:\Expert_Datasets\绿地识别\影像数据\150802临河区\150802GF2DOM01.tif
    output_fileDir = r"/output_path"  # r"/output_path" #sys.argv[2] #r"D:\Miscellaneous\picture\Output"  #r"E:\data\projects_and_topic\LULC_HuangHua\pred_patch_GuiXi"
    config_file = r"/workspace/configs/rsipac_segformer_test.yaml"
    cfg.update_from_file(config_file)
    cfg.PHASE = 'test'
    cfg.check_and_freeze()

    extraction = BigDataExtraction(input_fileDir, output_fileDir)
    extraction.extract_BigImage()


