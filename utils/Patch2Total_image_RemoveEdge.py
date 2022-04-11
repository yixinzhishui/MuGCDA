#分块读取
import os
import sys
import torch
import cv2
from osgeo import gdal
import numpy as np
import time
from tqdm import tqdm

from torchvision import transforms
import torch.nn as nn
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

Patch_Size = 3000
Stride = 2000

def set_batch_norm_attr(named_modules, attr, value):
    for m in named_modules:
        if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
            setattr(m[1], attr, value)

class GDAL_Image():
    def __init__(self, filename):
        self.image = gdal.Open(filename)
        if self.image == None:
            print(filename + "文件无法打开")

        self.img_width = self.image.RasterXSize
        self.img_height = self.image.RasterYSize

        self.img_geotrans = self.image.GetGeoTransform()
        self.img_proj = self.image.GetProjection()
        self.img_data = self.image.ReadAsArray(0, 0, 2, 2)
        self.get_RasterCount()
        #self.img_band = self.image.RasterCount
        #del image

    def get_RasterCount(self):
        if len(self.img_data.shape) == 2:
            self.img_bands = 1
        elif len(self.img_data.shape) == 3:
            self.img_bands = self.img_data.shape[0]

    def Get_ImageData_Patch(self, w, h, w_size, h_size):

        return self.image.ReadAsArray(int(w), int(h), int(w_size), int(h_size))

    def Set_ImageData(self):
        self.img_data = self.image.ReadAsArray(0, 0, self.img_width, self.img_height)

    @staticmethod
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

def get_ImageSlide_steps(image_height, image_width):
    height_steps = [h_steps if h_steps + Patch_Size < image_height else image_height - Patch_Size for h_steps in range(0, image_height, Stride)]
    width_steps = [w_steps if w_steps + Patch_Size < image_width else image_width - Patch_Size for w_steps in range(0, image_width, Stride)]
    height_steps = np.unique(height_steps)
    width_steps = np.unique(width_steps)
    return height_steps, width_steps

def get_EdgeCrop_sizes(height_steps, width_steps):
    patch_h_overlap = (np.array(height_steps) + Patch_Size)[:-1] - np.array(height_steps)[1:]
    patch_w_overlap = (np.array(width_steps) + Patch_Size)[:-1] - np.array(width_steps)[1:]
    h_top, h_bottom = list(map(lambda x: int(x), [0] + list(patch_h_overlap / 2))), list(map(lambda x: int(x), list(-1 * patch_h_overlap / 2) + [0]))
    w_left, w_right = list(map(lambda x: int(x), [0] + list(patch_w_overlap / 2))), list(map(lambda x: int(x), list(-1 * patch_w_overlap / 2) + [0]))

    return [h_top, h_bottom], [w_left, w_right]

def Get_datasetList(images_path):
    images_list = []

    for file in os.listdir(images_path):
        if file.endswith((".img")) or file.endswith((".tif")) or file.endswith((".png")):
            images_list.append(os.path.join(images_path, file))

    return images_list

def decode_segmap(mask):

    assert len(mask.shape) == 2, "the len of mask unexpect"
    height, width = mask.shape
    decode_mask = np.zeros((height, width, 3), dtype=np.uint)
    debug = cfg.DATASET.NUM_CLASSES
    for j in range(cfg.DATASET.NUM_CLASSES):       #range(self.config.num_classes):
        for k in range(3):
            decode_mask[:, :, k][np.where(mask[:, :] == j)] = cfg.DATASET.CLASS_INDEX[j][k]
    return decode_mask.astype(np.uint8)

def TotalImage_extraction(intput_file, output_dir, transform):
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    model = get_segmentation_model().to(args.device)

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
        #logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
        set_batch_norm_attr(model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.to(torch.device(args.device))
    model.eval()
    if os.path.isdir(intput_file):
        img_paths = [os.path.join(intput_file, x) for x in os.listdir(intput_file)]
    else:
        img_paths = [intput_file]
    for img_path in tqdm(img_paths, total=len(img_paths)):
        Image_ = GDAL_Image(img_path)

        image_data, image_width, image_height = Image_.img_data, Image_.img_width, Image_.img_height
        total_predict = np.zeros((image_height, image_width), dtype=np.float16)

        height_steps, width_steps = get_ImageSlide_steps(image_height, image_width)
        h_EdgeCropSize, w_EdgeCropSize = get_EdgeCrop_sizes(height_steps, width_steps)
        pass

        for ii, h_steps in tqdm(enumerate(height_steps), total=len(height_steps)):
            for jj, w_steps in enumerate(width_steps):
                #patch_image = image_data[:, h_steps:h_steps + Patch_Size, w_steps:w_steps + Patch_Size]
                patch_image = Image_.Get_ImageData_Patch(w_steps, h_steps, Patch_Size, Patch_Size)
                patch_image = cv2.cvtColor(np.rollaxis(patch_image, 0, 3), cv2.COLOR_BGR2RGB)
                patch_image = transform(Image.fromarray(patch_image)).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    output = model(patch_image)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                    #print(h_EdgeCropSize[0][ii], " ", h_EdgeCropSize[1][ii], " ", w_EdgeCropSize[0][jj], " ", w_EdgeCropSize[1][jj])
                    #print(h_steps + h_EdgeCropSize[0][ii], " ", h_steps + Patch_Size + h_EdgeCropSize[1][ii], " ", w_steps + w_EdgeCropSize[0][jj], " ", w_steps + Patch_Size + w_EdgeCropSize[1][jj])
                    total_predict[h_steps:h_steps + Patch_Size, w_steps:w_steps + Patch_Size] = pred
                    total_predict[h_steps + h_EdgeCropSize[0][ii]:h_steps + Patch_Size + h_EdgeCropSize[1][ii], w_steps + w_EdgeCropSize[0][jj]:w_steps + Patch_Size + w_EdgeCropSize[1][jj]] = pred[h_EdgeCropSize[0][ii]:Patch_Size + h_EdgeCropSize[1][ii], w_EdgeCropSize[0][jj]:Patch_Size + w_EdgeCropSize[1][jj]]
                    pass

        """彩色映射"""
        # TotalImage_DecodePredict = decode_segmap(total_predict)
        # TotalImage_DecodePredict = cv2.cvtColor(TotalImage_DecodePredict, cv2.COLOR_BGR2RGB)
        # TotalImage_DecodePredict = np.rollaxis(TotalImage_DecodePredict, 2, 0)
        # GDAL_Image.Write_Image(os.path.join(output_dir, os.path.basename(img_path)), TotalImage_DecodePredict, Image_.img_proj, Image_.img_geotrans)
        """直接写出"""
        GDAL_Image.Write_Image(os.path.join(output_dir, os.path.basename(img_path)), total_predict, Image_.img_proj, Image_.img_geotrans)

if __name__ == '__main__':
    input_fileDir = r"J:\Expert_Datasets\绿地识别\影像数据\500109北碚区\500109PL1+WV3+BJ2+GE1+GF2DOM01.img"  #E:\data\projects_and_topic\LULC_HuangHua\image_patch_GuiXi\360681BJ2+AP0DOM02_2_1.tif   #J:\Expert_Datasets\绿地识别\影像数据\150802临河区\150802GF2DOM01.tif
    output_fileDir = r"D:\Miscellaneous\picture\Output"  #r"E:\data\projects_and_topic\LULC_HuangHua\pred_patch_GuiXi"

    # output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'vis_result_{}_{}_{}_{}'.format(
    #     cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    TotalImage_extraction(input_fileDir, output_fileDir, transform)