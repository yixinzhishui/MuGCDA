from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import random
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import json
import pickle
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import umap

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
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        self.class_numbers = cfg.DATASET.NUM_CLASSES
        # dataset and dataloader
        self.test_dataset = get_segmentation_dataset(cfg.VAL.DATASET_NAME,
                                                        root='/data_zs/data/domain_adaptation/vaihingen_postdam/postdam',
                                                        data_list_root='/data_zs/data/domain_adaptation/vaihingen_postdam/postdam/split.csv',
                                                        split='train',
                                                        mode=cfg.DATASET.MODE,
                                                        transform=input_transform)
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


        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.preudo_label_dir, exist_ok=True)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.device)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_mean_vector_class(self, feat_cls, outputs, labels_val=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector_sample(self, feat_cls):
        vectors = []
        for n in range(feat_cls.size()[0]):
            s = feat_cls[n]
            s = F.adaptive_avg_pool2d(s, 1)
            s = s.squeeze().cpu().data.numpy()
            vectors.append(s)
        return vectors


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

        vectors, files = [], []
        for i, (images, targets, filename) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            images = images.to(self.device)
            # print(images.shape)
            # targets = targets.to(self.device)

            with torch.no_grad():
                output = model(images, get_feat=True)
                vectors_batch = self.calculate_mean_vector_sample(output['feat'])
                vectors.extend(vectors_batch)
                files.extend(filename)

        # Json = dict(source_smaple_feature_vector=vectors)
        with open(r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/source_smaple_feature_vector_potsdam_potsdam2vaihingen_onlysource_21file_0616.p', 'wb') as p_file:
            pickle.dump([vectors, files], p_file)

        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))


def get_subdir_label(path_list):
    sub_dir_list = []
    sub_dir_index = []
    label = []
    for path in path_list:
        sub_dir = path.split(os.path.sep)[-2]
        sub_dir_list.append(sub_dir)
        if sub_dir not in sub_dir_index:
            sub_dir_index.append(sub_dir)
    for dir in sub_dir_list:
        for ii, index in enumerate(sub_dir_index):
            if dir == index:
                label.append(ii)
                continue
    return label, sub_dir_index

if __name__ == '__main__':
    """导出特征向量到本地"""
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.test()

    """可视化特征向量"""
    # p_source_dirname = r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/source_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/source_smaple_feature_vector_potsdam_onlysource.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/source_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_21file.p'
    # p_target_dirname = r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/target_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_16file.p'  #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/target_smaple_feature_vector_vaihingen_onlysource.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/target_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_16file.p'
    # vectors_source, files_source = pickle.load(open(p_source_dirname, "rb"))
    # vectors_target, files_target = pickle.load(open(p_target_dirname, "rb"))
    #
    # # vectors_source = pickle.load(open(p_source_dirname, "rb"))
    # # vectors_target = pickle.load(open(p_target_dirname, "rb"))
    #
    # vectors_source = random.sample(vectors_source, len(vectors_target))
    # # vectors_target = random.sample(vectors_target, len(vectors_source))
    # vectors_source_label = [0] * len(vectors_source)
    # vectors_target_label = [1] * len(vectors_target)
    #
    # vectors = vectors_source + vectors_target #list(map(lambda x: x * 2, vectors_target))
    # vectors_concat = np.stack(vectors, axis=0)
    # vectors_label = np.array(vectors_source_label + vectors_target_label)
    #
    # reducer = umap.UMAP(random_state=42)
    # embedding = reducer.fit_transform(vectors_concat)
    #
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=5)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))  #boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # # plt.title('UMAP projection of the Digits dataset')
    # plt.axis('off')
    # plt.show()


    # p_source_dirname = r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/source_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_21file.p'
    # p_target_dirname = r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/target_smaple_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.5_16file.p'
    # vectors_source, files_source = pickle.load(open(p_source_dirname, "rb"))
    # vectors_target, files_target = pickle.load(open(p_target_dirname, "rb"))
    #
    # files = files_source + files_target
    # vectors_label, sub_dir_index = get_subdir_label(files)
    # # vectors_source = random.sample(vectors_source, len(vectors_target))
    # # vectors_target = random.sample(vectors_target, len(vectors_source))
    # # vectors_source_label = [0] * len(vectors_source)
    # # vectors_target_label = [1] * len(vectors_target)
    #
    # vectors = vectors_source + vectors_target #list(map(lambda x: x * 2, vectors_target))
    # vectors_concat = np.stack(vectors, axis=0)
    # vectors_label = np.array(vectors_label) #np.array(vectors_source_label + vectors_target_label)
    #
    # reducer = umap.UMAP(random_state=42)
    # embedding = reducer.fit_transform(vectors_concat)
    #
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=5)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(len(sub_dir_index) + 1) - 0.5).set_ticks(np.arange(len(sub_dir_index)))  #boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # plt.title('UMAP projection of the Digits dataset')
    # plt.show()
