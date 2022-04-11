import os
import sys
import torch
import cv2
from osgeo import gdal
import numpy as np
from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
import torch.nn as nn
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


def set_batch_norm_attr(named_modules, attr, value):
    for m in named_modules:
        if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
            setattr(m[1], attr, value)

def demo(in_path, out_path):
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
        #logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
        set_batch_norm_attr(model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                         device_ids=[args.local_rank], output_device=args.local_rank,
                                                         find_unused_parameters=True)
    model.to(torch.device(args.device))
    model.eval()

    if os.path.isdir(in_path):
        img_paths = [os.path.join(in_path, x) for x in os.listdir(in_path)]
    else:
        img_paths = [in_path]
    for img_path in tqdm(img_paths):
        image = readImage(img_path)                          #Image.open(img_path).convert('RGB')
        images = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
        #mask = get_color_pallete(pred, cfg.DATASET.NAME)
        mask = decode_segmap(pred)
        cv2.imwrite(os.path.join(out_path, os.path.basename(img_path).replace('.tif', '.png')),
                    cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

def readImage(dirname):
    if dirname.endswith('.png'):
        image = cv2.imread(dirname, -1)
    elif dirname.endswith('.tif'):
        Img = gdal.Open(dirname)
        image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
        if len(image.shape) == 3:
            image = np.rollaxis(image, 0, 3)
    return image

def decode_segmap(mask):

    assert len(mask.shape) == 2, "the len of mask unexpect"
    height, width = mask.shape
    decode_mask = np.zeros((height, width, 3), dtype=np.uint)
    debug = cfg.DATASET.NUM_CLASSES
    for j in range(cfg.DATASET.NUM_CLASSES):  # range(self.config.num_classes):
        for k in range(3):
            decode_mask[:, :, k][np.where(mask[:, :] == j)] = cfg.DATASET.CLASS_INDEX[j][k]
    return decode_mask.astype(np.uint8)

if __name__ == '__main__':
    input_imgDir = r"/dataset/Open_Datasets/LULC/GID/img_dir/test"
    output_imgDir = r"/code/python/pytorch/runs/visuals"
    demo(input_imgDir, output_imgDir)

