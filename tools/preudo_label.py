from __future__ import print_function

import os
import sys
import numpy as np
import cv2

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
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume
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
        self.val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, data_list_root=cfg.DATASET.DATA_LIST, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        #self.classes = val_dataset.NUM_CLASS    #NUM_CLASS    val_dataset.classes
        #self.classes = ["湿地", "耕地", "种植园用地", "林地", "草地", "道路", "建筑物", "水体", "未利用地", "背景"]
        self.classes = ["背景", "耕地", "林地", "草地", "灌木林","湿地", "水体", "人造地表"]
        # self.classes = ["background", "industrial land", "urban residential", "rural residential", "traffic land",
        #                 "paddy field", "irrigated land", "dry cropland", "garden plot", "arbor woodland", "shrub land",
        #                 "natural grassland", "artificial grassland", "river", "lake", "pond"]
        # create network
        self.model = get_segmentation_model().to(self.device)
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

        self.output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, cfg.VISUAL.CURRENT_NAME)
        self.preudo_label_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'val_preudo_label_' + cfg.VISUAL.CURRENT_NAME)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.preudo_label_dir):
            os.makedirs(self.preudo_label_dir)

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


        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()

        predicted_label = np.zeros((len(self.val_dataset), cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
        predicted_prob = np.zeros((len(self.val_dataset), cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
        image_name = []

        for i, (images, targets, filename) in enumerate(self.val_dataset):
            images = images.unsqueeze(0).to(self.device)
            targets = torch.tensor(targets).unsqueeze(0).to(self.device)

            with torch.no_grad():
                #output = model.evaluate(image)
                output = model(images)
                output = torch.softmax(output, 1)
                pred_label, pred_prob = torch.argmax(output, 1).squeeze(0).cpu().data.numpy(), torch.max(output, 1)[0].squeeze(0).cpu().data.numpy()
                output = torch.argmax(output, 1)

                predicted_label[i] = pred_label.copy()
                predicted_prob[i] = pred_prob.copy()
                image_name.append(filename)
                #output = torch.argmax(output.long(), 1)
            self.metric.update(output, targets)
            #pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, category_iou, category_pixAcc = self.metric.get(return_category_iou=True)
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, IOU_class: {}, pixAcc_class: {}".format(
                i + 1, pixAcc * 100, mIoU * 100, category_iou, category_pixAcc))
            #debug = torch.argmax(output[0], 0)

            #pred = torch.argmax(output[-1], 0).squeeze(0).cpu().data.numpy()
            preds = output.cpu().data.numpy().astype(np.uint8)
            for index in range(preds.shape[0]):
                mask = self.decode_segmap(preds[index])
                # if not os.path.exists(os.path.join(self.output_dir, filename[index].split('/')[-2])):
                #     os.makedirs(os.path.join(self.output_dir, filename[index].split('/')[-2]))
                cv2.imwrite(os.path.join(self.output_dir, filename), mask)  #, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)   #.replace('.tif', '.png')   #, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        thred = []
        for ii in range(cfg.DATASET.NUM_CLASSES):
            x = predicted_prob[predicted_label == ii]
            if len(x) == 0:
                thred.append(0)
            else:
                x = np.sort(x)
                thred.append(x[np.int(np.round(len(x) * 0.66))])
        thred = np.array(thred)
        thred[thred > 0.9] = 0.9
        print(thred)

        for index in range(len(self.val_dataset)):

            pred_label, pred_prob = predicted_label[index], predicted_prob[index]
            for ii in range(cfg.DATASET.NUM_CLASSES):
                pred_label[(pred_prob < thred[ii]) * (pred_label == ii)] = 250

            output = np.array(pred_label, dtype=np.uint8)
            output = self.decode_segmap(output)
            cv2.imwrite(os.path.join(self.preudo_label_dir, image_name[index]), output)  # , cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  .replace('.tif', '.png')   #, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            #mask = get_color_pallete(pred, cfg.DATASET.NAME)
            #outname = os.path.splitext(filename[-1])[0] + '.png'
            # mask = Image.fromarray(pred)
            #mask.save(os.path.join(self.output_dir, outname))
        synchronize()
        pixAcc, mIoU, category_iou, category_pixAcc = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}, IOU_class: {}, pixAcc_class: {}'.format(
                pixAcc * 100, mIoU * 100, category_iou, category_pixAcc))

        headers = ['class id', 'class name', 'iou', 'pixAcc']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i], category_pixAcc[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))


    def decode_segmap(self, mask):

        assert len(mask.shape) == 2, "the len of mask unexpect"
        assert cfg.DATASET.NUM_CLASSES == len(
            cfg.DATASET.CLASS_INDEX), "the value of NUM_CLASSES do not equal the len of CLASS_INDEX"
        height, width = mask.shape

        if isinstance(cfg.DATASET.CLASS_INDEX[0], int):
            decode_mask = np.zeros((height, width), dtype=np.uint)

            for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
                decode_mask[mask == index] = pixel
            decode_mask[mask == 250] = 250
        else:
            decode_mask = np.zeros((height, width, 3), dtype=np.uint)
            for index, pixel in enumerate(cfg.DATASET.CLASS_INDEX):  # range(self.config.num_classes):
                decode_mask[mask == index] = pixel
            decode_mask[mask == 250] = [250, 250, 250]
        return decode_mask.astype(np.uint8)

if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    #cfg.ROOT_PATH = r'/data_zs/data/test_crop'             #root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
