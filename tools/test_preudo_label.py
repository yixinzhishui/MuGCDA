from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

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
        self.test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, root=cfg.VAL.ROOT_PATH, split='val', mode='test', transform=input_transform)
        #val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, data_list_root=cfg.DATASET.DATA_LIST, split='val', mode='testval', transform=input_transform)
        self.test_sampler = make_data_sampler(self.test_dataset, False, args.distributed)
        self.test_batch_sampler = make_batch_data_sampler(self.test_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                          batch_sampler=self.test_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        #self.classes = val_dataset.NUM_CLASS    #NUM_CLASS    val_dataset.classes
        #self.classes = ["湿地", "耕地", "种植园用地", "林地", "草地", "道路", "建筑物", "水体", "未利用地", "背景"]
        #self.classes = ["背景", "耕地", "林地", "草地", "灌木林","湿地", "水体", "人造地表"]
        # self.classes = ["background", "industrial land", "urban residential", "rural residential", "traffic land",
        #                 "paddy field", "irrigated land", "dry cropland", "garden plot", "arbor woodland", "shrub land",
        #                 "natural grassland", "artificial grassland", "river", "lake", "pond"]
        self.classes = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if cfg.TRAIN.MODEL_SCALE > 1:
            self.model = SegmentationScale(self.model, float(cfg.TRAIN.MODEL_SCALE))
            print("--------------------------model scale:{}".format(cfg.TRAIN.MODEL_SCALE))

            # resume checkpoint if needed
        self.model, _, _, _, _ = load_model_resume(self.model)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(cfg.DATASET.NUM_CLASSES , args.distributed)

        self.output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, cfg.VISUAL.CURRENT_NAME)
        self.preudo_label_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, cfg.VISUAL.CURRENT_NAME)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.preudo_label_dir):
            os.makedirs(self.preudo_label_dir)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def test(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model


        logging.info("Start test, Total sample: {:d}".format(len(self.test_dataset)))
        import time
        time_start = time.time()

        predicted_label = np.zeros((len(self.test_dataset), cfg.VAL.CROP_SIZE, cfg.VAL.CROP_SIZE))
        predicted_prob = np.zeros((len(self.test_dataset), cfg.VAL.CROP_SIZE, cfg.VAL.CROP_SIZE))
        image_name = []

        for i, (images, filename) in tqdm(enumerate(self.test_dataset), total=len(self.test_dataset)):
            images = images.unsqueeze(0).to(self.device)

            with torch.no_grad():
                #output = model.evaluate(image)
                output = model(images)
                output = torch.softmax(output, 1)
                #output = torch.argmax(output, 1)
                pred_label, pred_prob = torch.argmax(output, 1).squeeze(0).cpu().data.numpy(), torch.max(output, 1)[0].squeeze(0).cpu().data.numpy()
                output = torch.argmax(output, 1)

            predicted_label[i] = pred_label.copy()
            predicted_prob[i] = pred_prob.copy()
            image_name.append(filename)

            #pred = torch.argmax(output[-1], 0).squeeze(0).cpu().data.numpy()
            preds = output.cpu().data.numpy().astype(np.uint8)
            for index in range(preds.shape[0]):
                mask = self.decode_segmap(preds[index])
                # if not os.path.exists(os.path.join(self.output_dir, filename[index].split('/')[-2])):
                #     os.makedirs(os.path.join(self.output_dir, filename[index].split('/')[-2]))
                cv2.imwrite(os.path.join(self.output_dir, filename), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))   #.replace('.tif', '.png')   #, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

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

        for index in tqdm(range(len(self.test_dataset)), total=len(self.test_dataset), desc="get pesudo label"):

            pred_label, pred_prob, img_name = predicted_label[index], predicted_prob[index], image_name[index]

            # for ii in range(cfg.DATASET.NUM_CLASSES):
            #     pred_label[(pred_prob < thred[ii]) * (pred_label == ii)] = 255

            output = np.array(pred_label, dtype=np.uint8)
            output = self.decode_segmap(output)
            cv2.imwrite(os.path.join(self.preudo_label_dir, img_name), output)

        synchronize()
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))


    def decode_segmap(self, mask):

        assert len(mask.shape) == 2, "the len of mask unexpect"

        height, width = mask.shape

        decode_mask = 255 * np.ones((height, width), dtype=np.uint8)

        for  pixel, index in cfg.DATASET.CLASS_INDEX[0].items():  # range(self.config.num_classes):
            decode_mask[mask == index] = pixel

        return decode_mask.astype(np.uint8)

if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.test()
