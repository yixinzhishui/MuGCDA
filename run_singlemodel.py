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



class SegmentationInterpret_BigData():
    def __init__(self, filename, patch_size=512, stride=300, transform=None, rw_mode="cv", patch_auto=True, max_patch_size=2600, **kwargs):
        self.filename = filename
        self.transform = transform
        self.rw_mode = rw_mode
        self.max_patch_size = max_patch_size

        if rw_mode == "gdal":
            self.image = gdal.Open(filename)
            self.img_width = self.image.RasterXSize
            self.img_height = self.image.RasterYSize

            self.img_geotrans = self.image.GetGeoTransform()
            self.img_proj = self.image.GetProjection()
            self.get_RasterCount()
        else:
            self.img_data = cv2.imread(filename, -1)[:, :, 0:3]
            #self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2RGB)
            #self.img_data = np.rollaxis(self.img_data, 2, 0)
            self.img_height, self.img_width, self.img_bands = self.img_data.shape

        if self.img_height * self.img_width < max_patch_size * max_patch_size:
            self.model_input_one = True
        else:
            self.model_input_one = False

        self.pred_big_data = np.zeros((self.img_height, self.img_width), dtype=np.uint8)  #np.float16


        if patch_auto:
            self.h_patch_size, self.w_patch_size, self.h_stride, self.w_stride = self.get_patch_stride(self.img_height, self.img_width, max_patch_size)
            if self.model_input_one:
                self.img_index_list = self.get_PatchIndexList(self.img_height, self.img_width)
            else:
                self.img_index_list = self.get_PatchIndexList(self.img_height, self.img_width, self.h_patch_size, self.w_patch_size, self.h_stride, self.w_stride)
        else:
            self.h_patch_size = patch_size
            self.w_patch_size = patch_size
            self.h_stride = stride
            self.w_stride = stride
            self.img_index_list = self.get_PatchIndexList(self.img_height, self.img_width, self.h_patch_size, self.w_patch_size, self.h_stride, self.w_stride)


    def get_patch_stride(self, img_height, img_width, max_patch_size):
        h_patch_size, w_patch_size, h_stride, w_stride = 0, 0, 0, 0
        if img_height < max_patch_size or img_width < max_patch_size:
            if img_height < max_patch_size:
                h_patch_size = self.img_height
                w_patch_size = h_patch_size #h_patch_size #int(2048 * 2048 / h_patch_size) #int(max_patch_size * max_patch_size / h_patch_size)
                h_stride = int(h_patch_size - h_patch_size // 3) #int(h_patch_size - 256)
                w_stride = int(w_patch_size - w_patch_size // 3) #int(w_patch_size - 256)
            if self.img_width < max_patch_size:
                w_patch_size = self.img_width
                h_patch_size = w_patch_size #w_patch_size #int(2048 * 2048 / w_patch_size) #int(max_patch_size * max_patch_size / w_patch_size)
                w_stride = int(w_patch_size - w_patch_size // 3) #int(w_patch_size - 256)
                h_stride = int(h_patch_size - h_patch_size // 3) #int(h_patch_size - 256)
        else:
            h_patch_size = max_patch_size #2048
            w_patch_size = max_patch_size
            h_stride = int(h_patch_size - h_patch_size // 3) #int(h_patch_size - 256) #1024   #int(h_patch_size / 2) #1024
            w_stride = int(w_patch_size - w_patch_size // 3) #int(w_patch_size - 256)  # 1024  int(w_patch_size / 2)

        return h_patch_size, w_patch_size, h_stride, w_stride

    def get_PatchIndexList(self, img_height, img_width, h_patch_size=None, w_patch_size=None, h_stride=None, w_stride=None):
        # patch_size_ = None
        # stride_ = None
        img_index_list = []

        def get_ImageSlide_steps(image_height, image_width, h_patch_size, w_patch_size, h_stride, w_stride):

            height_steps = [h_steps if h_steps + h_patch_size < image_height else image_height - h_patch_size for h_steps in
                            range(0, image_height, h_stride)]
            width_steps = [w_steps if w_steps + w_patch_size < image_width else image_width - w_patch_size for w_steps in
                           range(0, image_width, w_stride)]
            height_steps = np.unique(height_steps)
            width_steps = np.unique(width_steps)
            return height_steps, width_steps

        def get_EdgeCrop_sizes(height_steps, width_steps, h_patch_size, w_patch_size):
            patch_h_overlap = (np.array(height_steps) + h_patch_size)[:-1] - np.array(height_steps)[1:]
            patch_w_overlap = (np.array(width_steps) + w_patch_size)[:-1] - np.array(width_steps)[1:]
            h_top, h_bottom = list(map(lambda x: int(x), [0] + list(patch_h_overlap / 2))), list(
                map(lambda x: int(x), list(-1 * patch_h_overlap / 2) + [0]))
            w_left, w_right = list(map(lambda x: int(x), [0] + list(patch_w_overlap / 2))), list(
                map(lambda x: int(x), list(-1 * patch_w_overlap / 2) + [0]))

            return [h_top, h_bottom], [w_left, w_right]

        if h_patch_size is None or w_patch_size is None or h_stride is None or w_stride is None:
            img_index_list.append([0, 0, [0, 0], [0, 0]])
        else:
            height_steps, width_steps = get_ImageSlide_steps(img_height, img_width, h_patch_size, w_patch_size, h_stride, w_stride)
            h_EdgeCropSize, w_EdgeCropSize = get_EdgeCrop_sizes(height_steps, width_steps, h_patch_size, w_patch_size)

            for ii, h_step in enumerate(height_steps):
                for jj, w_step in enumerate(width_steps):
                    img_index_list.append([h_step, w_step, [h_EdgeCropSize[0][ii], h_EdgeCropSize[1][ii]],
                                           [w_EdgeCropSize[0][jj], w_EdgeCropSize[1][jj]]])

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
        if self.model_input_one:
            patch_image = self.img_data
        else:
            patch_image = self.img_data[h_step:h_step + self.h_patch_size, w_step:w_step + self.w_patch_size, :]
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

        h_steps, w_steps, h_EdgeCropSizes, w_EdgeCropSizes = image_indexs

        for item in range(images.shape[0]):
            if self.model_input_one:
                self.pred_big_data = images[item]
            else:
                h_step, w_step, h_EdgeCropSize, w_EdgeCropSize = h_steps[item], w_steps[item], \
                                                                 [h_EdgeCropSizes[0][item], h_EdgeCropSizes[1][item]], \
                                                                 [w_EdgeCropSizes[0][item], w_EdgeCropSizes[1][item]]
                image = images[item]
                # self.pred_big_data[h_step:h_step + Patch_Size, w_step:w_step + Patch_Size] = image
                self.pred_big_data[h_step + h_EdgeCropSize[0]:h_step + self.h_patch_size + h_EdgeCropSize[1],
                w_step + w_EdgeCropSize[0]:w_step + self.w_patch_size + w_EdgeCropSize[1]] = image[h_EdgeCropSize[
                                                                                                     0]:self.h_patch_size +
                                                                                                        h_EdgeCropSize[
                                                                                                            1],
                                                                                           w_EdgeCropSize[
                                                                                               0]:self.w_patch_size +
                                                                                                  w_EdgeCropSize[1]]
        # for item, image_index in enumerate(image_indexs):
        #     print(image_index)
        #     h_step, w_step, h_EdgeCropSize, w_EdgeCropSize = image_index
        #     image = images[item]
        #     #self.pred_big_data[h_step:h_step + Patch_Size, w_step:w_step + Patch_Size] = image
        #     self.pred_big_data[h_step + h_EdgeCropSize[0]:h_step + self.patch_size + h_EdgeCropSize[1], w_step + w_EdgeCropSize[0]:w_step + self.patch_size + w_EdgeCropSize[1]] = image[h_EdgeCropSize[0]:self.patch_size + h_EdgeCropSize[1], w_EdgeCropSize[0]:self.patch_size + w_EdgeCropSize[1]]

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
            save_path = os.path.join(filename, os.path.basename(self.filename).replace('.tif', '.png'))
        elif os.path.isfile(filename):
            save_path = filename.replace('.tif', '.png')
        if mode == "gdal":
            Write_Image(save_path, save_image, img_proj=self.img_proj, img_geotrans=self.img_geotrans)
        else:
            cv2.imwrite(save_path, save_image)



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

        self.ttach_transform = ttach.aliases.hflip_transform()

        self.model = get_segmentation_model().to(self.device)
        self.model, _, _, _, _ = load_model_resume(self.model)
        print(cfg.TEST.TEST_MODEL_PATH)
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

        return decode_mask

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def extract_BigImage(self):
        self.model.eval()
        # if self.args.distributed:
        #     model = self.model.module
        # else:
        model = self.model
        time_start = time.time()
        for filename in tqdm(self.input_pathList):
            #torch.cuda.empty_cache()
            img_dataset = SegmentationInterpret_BigData(filename, patch_size=512, stride=300, transform=self.input_transform)
            img_patch_loader = data.DataLoader(img_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.DATASET.WORKERS, pin_memory=True)
            print("width:{}".format(img_dataset.img_width))
            print("height:{}".format(img_dataset.img_height))
            print("filename:{}".format(img_dataset.filename))
            print("h_patch:{}".format(img_dataset.h_patch_size))
            print("w_patch:{}".format(img_dataset.w_patch_size))
            print("h_stride:{}".format(img_dataset.h_stride))
            print("w_stride:{}".format(img_dataset.w_stride))
            for ii, (images, images_index) in tqdm(enumerate(img_patch_loader), total=len(img_patch_loader)):
                images = images.to(self.device)
                with torch.no_grad():
                    output = None
                    count = 0
                    # output = model.evaluate(image)
                    if cfg.TEST.USE_TTA:

                        # model = ttach.SegmentationTTAWrapper(model, ttach.aliases.hflip_transform(), merge_mode="mean")

                        for transform in self.ttach_transform:
                            augmented_image = transform.augment_image(images)
                            model_output = model(augmented_image)
                            deaug_output = transform.deaugment_mask(model_output)
                            # preds_aug.append(deaug_output)
                            if output is None:
                                output = deaug_output
                            else:
                                output += deaug_output
                            count += 1
                    if output is None:
                        output = model(images)
                    else:
                        output += model(images)
                    count += 1
                    print(count)

                    output = torch.argmax(output, 1)

                    preds = output.cpu().data.numpy().astype(np.uint8)
                    img_dataset.set_BigData(preds, images_index)

            pred_bigdata = img_dataset.get_BigData()
            pred_bigdata = self.decode_segmap(pred_bigdata)
            img_dataset.write_BigData(self.output_path, pred_bigdata)
            #img_dataset.write_BigData(self.output_path)


if __name__ == '__main__':
    input_fileDir = r"/input_path" #"/data/data_tmp" #"/input_path"  # r"/input_path" #sys.argv[1] #r"J:\Expert_Datasets\绿地识别\影像数据\500109北碚区\500109PL1+WV3+BJ2+GE1+GF2DOM01.img"  #E:\data\projects_and_topic\LULC_HuangHua\image_patch_GuiXi\360681BJ2+AP0DOM02_2_1.tif   #J:\Expert_Datasets\绿地识别\影像数据\150802临河区\150802GF2DOM01.tif
    output_fileDir = r"/output_path"  # r"/output_path" #sys.argv[2] #r"D:\Miscellaneous\picture\Output"  #r"E:\data\projects_and_topic\LULC_HuangHua\pred_patch_GuiXi"
    config_file = r"/workspace/configs/rsipac_segformer_test.yaml"
    cfg.update_from_file(config_file)
    cfg.PHASE = 'test'
    cfg.check_and_freeze()

    extraction = BigDataExtraction(input_fileDir, output_fileDir)
    extraction.extract_BigImage()


