import os
import sys
import torch
import cv2

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

if __name__ == "__main__":
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.check_and_freeze()
    default_setup(args)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model()
    model.eval()
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
        # logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
        set_batch_norm_attr(model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

    #scripted_module = torch.jit.script(model)
    model.encoder.set_swish(memory_efficient=False)
    scripted_module = torch.jit.trace(model, torch.rand(1, 6, 1024, 1024))
    scripted_module.save(r"/data/data_zs/0721/landcover_inference/landcover_1.pt")
