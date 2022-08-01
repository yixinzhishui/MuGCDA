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
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
import random
import pandas as pd

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
        self.test_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                                        root='/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/visual/debug_potsdam_test', #debug #debug_potsdam_test  #/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/val   #'/data_zs/data/domain_adaptation/vaihingen_postdam/postdam'
                                                        data_list_root=None,
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
        print('-----------------feat_cls:{}'.format(feat_cls.shape))
        print('-----------------outputs_pred:{}'.format(outputs_pred.shape))
        for n in range(feat_cls.size()[0]):
            count = 0
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                # if (outputs_pred[n][t] > 0).sum() < 10:
                #     continue
                print('-----------------s:{}'.format(outputs_pred[n][t].shape))
                s = feat_cls[n] * outputs_pred[n][t] / scale_factor[n][t]
                print('-----------------s:{}'.format(s.shape))
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                # s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                s = s.flatten(1).transpose(0, 1).squeeze().cpu().data.numpy()
                delete_row = np.all(s == [0] * s.shape[1], 1)
                print('-----------------delete_row:{}'.format(delete_row.shape))
                delete_row_index = np.where(delete_row == True)[0]
                print('-----------------s_squeeze:{}'.format(s.shape))
                s = np.delete(s, delete_row_index.tolist(), axis=0)
                count += s.shape[0]
                print('-----------------len_index:{}'.format(len(delete_row_index)))
                print('-----------------index:{}'.format(delete_row_index))
                print('-----------------debug:{}'.format(delete_row.shape))
                print('-----------------s_squeeze_delete:{}'.format(s.shape))
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(np.array([t] * s.shape[0]))
            print('-----------------count:{}'.format(count))
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

        vectors, ids, files = [], [], []
        for i, (images, targets, filename) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            images = images.to(self.device)
            # print(images.shape)
            # targets = targets.to(self.device)

            with torch.no_grad():
                output = model(images, get_feat=True)
                vectors_batch, ids_batch = self.calculate_mean_vector_class(output['feat'], output['out'])
                vectors.extend(vectors_batch)
                ids.extend(ids_batch)
                files.extend(filename)

        # Json = dict(source_smaple_feature_vector=vectors)
        with open(r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_onlysource_source_0.44085.p', 'wb') as p_file:  #deeplabv2_class_feature_vector_potsdam2vaihingen_onlytarget.p  #/data_zs/output/vaihingen2potsdam/config/feature_class   /data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p
            pickle.dump([vectors, ids, files], p_file)

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
    # args = parse_args()
    # cfg.update_from_file(r'/data_zs/code/loveDA/pytorchAI_segmentation_loveda/configs/test/potsdam2vaihingen_onlysource_deeplabv2.yaml')
    # cfg.update_from_list(args.opts)
    # cfg.PHASE = 'test'
    # cfg.check_and_freeze()
    #
    # default_setup(args)
    #
    # evaluator = Evaluator(args)
    # evaluator.test()

    """可视化特征向量"""
    # p_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_uda_source.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/target_class_feature_vector_vaihingen_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p'
    # vectors, ids, files = pickle.load(open(p_dirname, "rb"))
    #
    # # vectors_target = random.sample(vectors_target, len(vectors_source))
    #
    # vectors_concat = np.concatenate(vectors, axis=0)
    # vectors_label = np.concatenate(ids, axis=0)
    #
    # print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    # print('-----------------vectors_label:{}'.format(vectors_label.shape))
    # reducer = umap.UMAP(n_neighbors=100, random_state=42, n_epochs=1000)   #https://cloud.tencent.com/developer/article/1901726
    # embedding = reducer.fit_transform(vectors_concat)
    #
    # # reducer = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=1500)
    # # embedding = reducer.fit_transform(vectors_concat)
    #
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=6)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(len(np.unique(vectors_label)) + 1) - 0.5).set_ticks(np.arange(len(np.unique(vectors_label))))  #boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
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


    '''可视化特征向量---指定两个类'''
    # p_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_uda_target_0.57631.p'  # r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/target_class_feature_vector_vaihingen_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p'
    # vectors, ids, files = pickle.load(open(p_dirname, "rb"))
    #
    # # vectors_target = random.sample(vectors_target, len(vectors_source))
    #
    # vectors_concat = np.concatenate(vectors, axis=0)
    # vectors_label = np.concatenate(ids, axis=0)
    #
    # # select_index = np.where((vectors_label == 1) | (vectors_label == 2) | (vectors_label == 3))[0]
    # select_index = np.where((vectors_label == 2) | (vectors_label == 3))[0]
    # # print(len(select_index))
    # # select_index = select_index[np.random.choice(len(select_index), int(len(select_index) / 16))]
    # # print(len(select_index))
    # vectors_concat = vectors_concat[select_index, :]
    # vectors_label = vectors_label[select_index]
    #
    # # vectors_label[np.where(vectors_label == 1)] = 0
    # # vectors_label[np.where(vectors_label == 2)] = 1
    # # vectors_label[np.where(vectors_label == 3)] = 2
    #
    # vectors_label[np.where(vectors_label == 2)] = 0
    # vectors_label[np.where(vectors_label == 3)] = 1
    #
    # print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    # print('-----------------vectors_label:{}'.format(vectors_label.shape))
    # reducer = umap.UMAP(n_neighbors=100, random_state=42, n_epochs=1000)   #https://cloud.tencent.com/developer/article/1901726
    # embedding = reducer.fit_transform(vectors_concat)
    #
    # # reducer = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=2500)
    # # embedding = reducer.fit_transform(vectors_concat)
    #
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=6)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(len(np.unique(vectors_label)) + 1) - 0.5).set_ticks(
    #     np.arange(len(np.unique(vectors_label))))  # boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # # plt.title('UMAP projection of the Digits dataset')
    # # plt.axis('off')
    # plt.show()
    #
    # out_csv = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_uda_target_0.57631.csv'
    # dict = {'x': embedding[:, 0].tolist(),
    #         'y': embedding[:, 1].tolist(),
    #         'label': vectors_label.tolist()}
    # df = pd.DataFrame(dict)
    # df.to_csv(out_csv)


    '''可视化源域和目标域的特征向量---指定两个类'''
    # source_p_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_onlysourcce_source_0.42789.p'  # r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/target_class_feature_vector_vaihingen_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p'
    # target_p_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_onlysourcce_target_0.42789.p'
    # vectors_source, ids_source, files_source = pickle.load(open(source_p_dirname, "rb"))
    # vectors_target, ids_target, files_target = pickle.load(open(target_p_dirname, "rb"))
    #
    # # vectors_target = random.sample(vectors_target, len(vectors_source))
    #
    # vectors_concat_source = np.concatenate(vectors_source, axis=0)
    # vectors_label_source = np.concatenate(ids_source, axis=0)
    # vectors_concat_target = np.concatenate(vectors_target, axis=0)
    # vectors_label_target = np.concatenate(ids_target, axis=0)
    #
    # # select_index = np.where((vectors_label == 1) | (vectors_label == 2) | (vectors_label == 3))[0]
    # select_index_source = np.where((vectors_label_source == 2) | (vectors_label_source == 3))[0]
    # vectors_concat_source = vectors_concat_source[select_index_source, :]
    # vectors_label_source = vectors_label_source[select_index_source]
    #
    # select_index_target = np.where((vectors_label_target == 2) | (vectors_label_target == 3))[0]
    # vectors_concat_target = vectors_concat_target[select_index_target, :]
    # vectors_label_target = vectors_label_target[select_index_target]
    #
    # # vectors_label[np.where(vectors_label == 1)] = 0
    # # vectors_label[np.where(vectors_label == 2)] = 1
    # # vectors_label[np.where(vectors_label == 3)] = 2
    #
    # vectors_label_source[np.where(vectors_label_source == 2)] = 0
    # vectors_label_source[np.where(vectors_label_source == 3)] = 1
    # vectors_label_target[np.where(vectors_label_target == 2)] = 2
    # vectors_label_target[np.where(vectors_label_target == 3)] = 3
    #
    # print('-----------------vectors_concat_source:{}'.format(vectors_concat_source.shape))
    # print('-----------------vectors_label_source:{}'.format(vectors_label_source.shape))
    # print('-----------------vectors_concat_target:{}'.format(vectors_concat_target.shape))
    # print('-----------------vectors_label_target:{}'.format(vectors_label_target.shape))
    #
    # vectors_concat = np.concatenate([vectors_concat_source, vectors_concat_target], axis=0)
    # vectors_label = np.concatenate([vectors_label_source, vectors_label_target], axis=0)
    # print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    # print('-----------------vectors_label:{}'.format(vectors_label.shape))
    #
    # reducer = umap.UMAP(n_neighbors=100, random_state=42, n_epochs=1000)   #https://cloud.tencent.com/developer/article/1901726
    # embedding = reducer.fit_transform(vectors_concat)
    #
    # # reducer = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=2500)
    # # embedding = reducer.fit_transform(vectors_concat)
    #
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=6)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(len(np.unique(vectors_label)) + 1) - 0.5).set_ticks(
    #     np.arange(len(np.unique(vectors_label))))  # boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # # plt.title('UMAP projection of the Digits dataset')
    # # plt.axis('off')
    # plt.show()


    '''可视化源域和目标域的特征向量---指定两个类-----fromcsv'''
    # source_csv_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_uda_source_0.57631.csv'  # r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/target_class_feature_vector_vaihingen_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p'
    # target_csv_dirname = r'/data_zs/config/domain_adaption/deeplabv2_class_feature_vector_potsdam2vaihingen_uda_target_0.57631.csv'
    #
    # embedding_source = pd.read_csv(source_csv_dirname)
    # embedding_source_x = embedding_source['x']
    # embedding_source_y = embedding_source['y']
    # embedding_source_label = embedding_source['label']
    #
    # embedding_source_x = np.array(embedding_source_x)
    # embedding_source_x = embedding_source_x - 10
    # embedding_source_y = np.array(embedding_source_y)
    # embedding_source_y = embedding_source_y - 10
    # embedding_source_label = np.array(embedding_source_label)
    #
    #
    # embedding_target = pd.read_csv(target_csv_dirname)
    # embedding_target_x = embedding_target['x']
    # embedding_target_y = embedding_target['y']
    # embedding_target_label = embedding_target['label']
    #
    # embedding_target_x = np.array(embedding_target_x)
    # embedding_target_x = embedding_target_x #- 50
    # embedding_target_y = np.array(embedding_target_y)
    # embedding_target_y = embedding_target_y #+ 40
    # embedding_target_label = np.array(embedding_target_label)
    # embedding_target_label[np.where(embedding_target_label == 0)] = 2
    # embedding_target_label[np.where(embedding_target_label == 1)] = 3
    #
    # embedding_x = np.concatenate([embedding_source_x, embedding_target_x], axis=0)
    # embedding_y = np.concatenate([embedding_source_y, embedding_target_y], axis=0)
    # embedding_label = np.concatenate([embedding_source_label, embedding_target_label], axis=0)
    #
    # # reducer = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=2500)
    # # embedding = reducer.fit_transform(vectors_concat)
    #
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(embedding_x, embedding_y, c=embedding_label, cmap='Spectral', s=6)
    # plt.gca().set_aspect('equal', 'datalim')
    # # plt.colorbar(boundaries=np.arange(len(np.unique(embedding_label)) + 1) - 0.5).set_ticks(
    # #     np.arange(len(np.unique(embedding_label))))  # boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # # plt.title('UMAP projection of the Digits dataset')
    # # plt.axis('off')
    #
    # ax = plt.subplot()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()

    # out_csv = r'/data_zs/config/domain_adaption/segformerb2_class_feature_vector_potsdam2vaihingen_uda_sourcetarget-5_0.66696.csv'
    # dict = {'x': embedding_x.tolist(),
    #         'y': embedding_y.tolist(),
    #         'label': embedding_label.tolist()}
    # df = pd.DataFrame(dict)
    # df.to_csv(out_csv)

    """可视化特征向量--全类(除去背景类)--from.p文件"""
    p_dirname = r'/data_zs/config/domain_adaption/segformerb2_class_feature_vector_potsdam2vaihingen_uda_target_0.66696.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/source_class_feature_vector_potsdam_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_21file.p' #r'/data_zs/output/potsdam2vaihingen/pytorchAI_segmentation/config/feature_class/target_class_feature_vector_vaihingen_potsdam2vaihingen_pesudo0.5_weight_st_online_ema_spatial-d5-w0.1-ema-source_16file.p'
    vectors, ids, files = pickle.load(open(p_dirname, "rb"))

    # vectors_target = random.sample(vectors_target, len(vectors_source))

    vectors_concat = np.concatenate(vectors, axis=0)
    vectors_label = np.concatenate(ids, axis=0)
    print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    print('-----------------vectors_label:{}'.format(vectors_label.shape))

    delete_index = np.where(vectors_label == 5)[0]
    print(len(delete_index))
    vectors_concat = np.delete(vectors_concat, delete_index, axis=0)
    vectors_label = np.delete(vectors_label, delete_index, axis=0)
    print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    print('-----------------vectors_label:{}'.format(vectors_label.shape))

    select_index = np.random.choice(len(vectors_label), int(len(vectors_label) / 16))
    print(len(select_index))
    vectors_concat = vectors_concat[select_index, :]
    vectors_label = vectors_label[select_index]

    print('-----------------vectors_concat:{}'.format(vectors_concat.shape))
    print('-----------------vectors_label:{}'.format(vectors_label.shape))


    reducer = umap.UMAP(n_neighbors=100, random_state=42, n_epochs=1000)   #https://cloud.tencent.com/developer/article/1901726
    embedding = reducer.fit_transform(vectors_concat)

    # reducer = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=1500)
    # embedding = reducer.fit_transform(vectors_concat)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=vectors_label, cmap='Spectral', s=6)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(len(np.unique(vectors_label)) + 1) - 0.5).set_ticks(np.arange(len(np.unique(vectors_label))))  #boundaries=np.arange(11) - 0.5).set_ticks(np.arange(2)
    # plt.title('UMAP projection of the Digits dataset')
    # plt.axis('off')
    # plt.show()

    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    out_csv = r'/data_zs/config/domain_adaption/segformerb2_class_feature_vector_potsdam2vaihingen_uda_target_0.66696_allclass.csv'
    dict = {'x': embedding[:, 0].tolist(),
            'y': embedding[:, 1].tolist(),
            'label': vectors_label.tolist()}
    df = pd.DataFrame(dict)
    df.to_csv(out_csv)

