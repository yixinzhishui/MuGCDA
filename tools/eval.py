from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import math
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume, SegmentationScale
import segmentron.solver.ttach as ttach

from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import get_color_pallete

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, data_list_root=cfg.VAL.DATA_LIST, split='val', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, data_list_root=cfg.DATASET.DATA_LIST, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        #self.classes = val_dataset.NUM_CLASS    #NUM_CLASS    val_dataset.classes
        #self.classes = ["湿地", "耕地", "种植园用地", "林地", "草地", "道路", "建筑物", "水体", "未利用地", "背景"]
        self.classes = ["耕地", "林地", "草地", "水域", "城乡", "未利用地"]
        # self.classes = ["background", "industrial land", "urban residential", "rural residential", "traffic land",
        #                 "paddy field", "irrigated land", "dry cropland", "garden plot", "arbor woodland", "shrub land",
        #                 "natural grassland", "artificial grassland", "river", "lake", "pond"]
        # create network
        self.model = get_segmentation_model().to(self.device)
        if cfg.TRAIN.MODEL_SCALE > 1:
            self.model = SegmentationScale(self.model, float(cfg.TRAIN.MODEL_SCALE))

        self.model, _, _, _, _ = load_model_resume(self.model)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(cfg.DATASET.NUM_CLASSES , args.distributed)

        self.output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'val_' + cfg.VISUAL.CURRENT_NAME)   #cfg.VISUAL.CURRENT_NAME
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        ttach_transforms = ttach.aliases.flip_transform()

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        for i, (images, targets, filename) in enumerate(self.val_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                output = None
                count = 0

                if cfg.TEST.USE_TTA:
                    #model = ttach.SegmentationTTAWrapper(model, ttach.aliases.hflip_transform(), merge_mode="mean")
                    for transform in ttach_transforms:
                        augmented_image = transform.augment_image(images)
                        model_output = model(augmented_image)
                        deaug_output = transform.deaugment_mask(model_output)
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
                if isinstance(output, (tuple, list)):
                    output = output[0]

                #output = torch.argmax(output.long(), 1)   #去掉long()
                output = torch.argmax(output, 1)
            self.metric.update(output, targets)
            #pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, category_iou, category_precision, category_recall, category_f1 = self.metric.get(return_category_iou=True)
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, IOU_class: {}, pixAcc_class: {}".format(
                i + 1, pixAcc * 100, mIoU * 100, category_iou, category_recall))
            #debug = torch.argmax(output[0], 0)

            #pred = torch.argmax(output[-1], 0).squeeze(0).cpu().data.numpy()
            preds = output.cpu().data.numpy().astype(np.uint8)
            for index in range(preds.shape[0]):
                mask = self.decode_segmap(preds[index])
                # if not os.path.exists(os.path.join(self.output_dir, filename[index].split('/')[-2])):
                #     os.makedirs(os.path.join(self.output_dir, filename[index].split('/')[-2]))
                cv2.imwrite(os.path.join(self.output_dir, filename[index]), mask)   #.replace('.tif', '.png')   #, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            #mask = get_color_pallete(pred, cfg.DATASET.NAME)
            #outname = os.path.splitext(filename[-1])[0] + '.png'
            # mask = Image.fromarray(pred)
            #mask.save(os.path.join(self.output_dir, outname))
        synchronize()
        pixAcc, mIoU, category_iou, category_precision, category_recall, category_f1 = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}, IOU_class: {}, pixAcc_class: {}'.format(
                pixAcc * 100, mIoU * 100, category_iou, category_recall))

        headers = ['class id', 'class name', 'iou', 'precision', 'recall', 'f1']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i], category_precision[i], category_recall[i], category_f1[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))


    # def decode_segmap(self, mask):
    #
    #     assert len(mask.shape) == 2, "the len of mask unexpect"
    #     assert cfg.DATASET.NUM_CLASSES == len(
    #         cfg.DATASET.CLASS_INDEX), "the value of NUM_CLASSES do not equal the len of CLASS_INDEX"
    #     height, width = mask.shape
    #
    #     if isinstance(cfg.DATASET.CLASS_INDEX[0], int):
    #         decode_mask = np.zeros((height, width), dtype=np.uint)
    #
    #         for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
    #             decode_mask[mask == index] = pixel
    #     else:
    #         decode_mask = np.zeros((height, width, 3), dtype=np.uint)
    #         for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
    #             decode_mask[mask == index] = pixel
    #
    #     return decode_mask.astype(np.uint8)

    def decode_segmap(self, mask):

        assert len(mask.shape) == 2, "the len of mask unexpect"
        assert cfg.DATASET.NUM_CLASSES == len(cfg.DATASET.CLASS_INDEX[0]), "the value of NUM_CLASSES do not equal the len of CLASS_INDEX"
        height, width = mask.shape

        decode_mask = np.zeros((height, width), dtype=np.uint)

        for pixel_value, class_index in cfg.DATASET.CLASS_INDEX[0].items():  # range(self.config.num_classes):
            decode_mask[mask == class_index] = pixel_value

        return decode_mask.astype(np.uint8)

if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    #cfg.ROOT_PATH = r"/data_zs/data/data_cj_images" #r'/data_zs/data/test_crop'             #root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
