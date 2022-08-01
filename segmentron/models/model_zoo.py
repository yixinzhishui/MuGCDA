import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from segmentron.utils.registry import Registry
import segmentation_models_pytorch as smp
from ..config import cfg

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""        #__doc__：https://blog.csdn.net/weixin_30885111/article/details/99468459   https://www.cnblogs.com/wenshinlee/p/12665089.html


class SegmentationScale(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            scale: float
    ):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):
        oldsize = x.shape[-1]
        x = F.interpolate(x, scale_factor=self.scale)
        x = self.model(x)
        x = F.interpolate(x, size=[oldsize, oldsize], mode="bilinear")
        return x


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    def load_backbone_pretrained(model):
        if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
            if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
                logging.info('Load backbone pretrained model from {}'.format(
                    cfg.TRAIN.BACKBONE_PRETRAINED_PATH
                ))

                pretrain_model_dict = torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH, map_location='cpu')
                load_model_dict = dict()
                # print(pretrain_model_dict.keys())
                # print( model.encoder.state_dict().keys())
                """当输出波段>3时"""
                for item, value in model.encoder.state_dict().items():
                    if item in pretrain_model_dict.keys() and value.shape == pretrain_model_dict[item].shape:
                        load_model_dict[item] = pretrain_model_dict[item]
                    else:
                        print(item)
                # for item, value in pretrain_model_dict.items():
                #     if item in model.encoder.state_dict().keys() and value.shape == model.encoder.state_dict()[item].shape:
                #         load_model_dict[item] = pretrain_model_dict[item]
                #     else:
                #         print(item)
                # load_model_dict["_fc.weight"] = 0    ##注：efficientnet：_fc.weight     _fc.bias   segmentation_models_pytorch内置有删除这两个key的代码
                # load_model_dict["_fc.bias"] = 0
                msg = model.encoder.load_state_dict(load_model_dict, strict=False)
                # msg = model.encoder.load_state_dict(pretrain_model_dict, strict=False)
                logging.info(msg)

    model_name = cfg.MODEL.MODEL_NAME
    if cfg.MODEL.SOURCE == "smp":
        ModelFramework = smp.__dict__[model_name]
        model = ModelFramework(encoder_name=cfg.MODEL.BACKBONE, in_channels=cfg.DATASET.NUM_CHANNELS, classes=cfg.DATASET.NUM_CLASSES, encoder_weights=None, activation=cfg.MODEL.ACTIVATION)
        #smp.Unet()
        load_backbone_pretrained(model)
    elif model_name == "HRNet_OCR":
        model = MODEL_REGISTRY.get(model_name)(cfg)
        model.init_weights(cfg.TRAIN.BACKBONE_PRETRAINED_PATH)
    else:
        model = MODEL_REGISTRY.get(model_name)()

    #load_model_pretrain(model)
    return model


# def load_model_pretrain(model):                   #加载模型，去某一层特定参数：https://www.jianshu.com/p/33bdf8ca82dd/
#     if cfg.PHASE == 'train':
#         if cfg.TRAIN.PRETRAINED_MODEL_PATH:
#             logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
#             pretrain_checkpoint_dict = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
#             state_dict_to_load = pretrain_checkpoint_dict['state_dict']
#             keys_wrong_shape = []
#             state_dict_suitable = OrderedDict()    #OrderedDict()：https://blog.csdn.net/u013066730/article/details/58120817
#             state_dict = model.state_dict()
#             for k, v in state_dict_to_load.items():
#                 if v.shape == state_dict[k].shape:
#                     state_dict_suitable[k] = v
#                 else:
#                     keys_wrong_shape.append(k)
#             logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
#             msg = model.load_state_dict(state_dict_suitable, strict=False)
#             logging.info(msg)
#
#             current_epoch = pretrain_checkpoint_dict['epoch']
#
#             return current_epoch
#     else:
#         if cfg.TEST.TEST_MODEL_PATH:
#             logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
#             msg = model.load_state_dict(torch.load(cfg.TEST.TEST_MODEL_PATH)['state_dict'], strict=False)  #
#
#             logging.info(msg)
#
#     return 0


def load_model_resume(model, optimizer=None, lr_scheduler=None, scaler=None, resume=None):  # 加载模型，去某一层特定参数：https://www.jianshu.com/p/33bdf8ca82dd/
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH or resume:
            if resume:
                logging.info('load pretrained model from {}'.format(resume))
                resume_checkpoint_dict = torch.load(resume, map_location=torch.device('cpu'))
            else:
                logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
                resume_checkpoint_dict = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH, map_location=torch.device('cpu'))

            state_dict_to_load = resume_checkpoint_dict['state_dict']
            keys_wrong_shape = []
            state_dict_suitable = OrderedDict()  # OrderedDict()：https://blog.csdn.net/u013066730/article/details/58120817
            state_dict = model.state_dict()
            for k, v in state_dict_to_load.items():
                if v.shape == state_dict[k].shape:
                    state_dict_suitable[k] = v
                else:
                    keys_wrong_shape.append(k)
            logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
            msg = model.load_state_dict(state_dict_suitable, strict=False)
            logging.info(msg)

            # if optimizer is not None and resume_checkpoint_dict['optimizer'] is not None:
            #     logging.info('resume optimizer from resume state..')
            #     optimizer.load_state_dict(resume_checkpoint_dict['optimizer'])
            # if lr_scheduler is not None and resume_checkpoint_dict['lr_scheduler'] is not None:
            #     logging.info('resume lr scheduler from resume state..')
            #     lr_scheduler.load_state_dict(resume_checkpoint_dict['lr_scheduler'])
            # if scaler is not None and resume_checkpoint_dict['scaler'] is not None:
            #     logging.info('resume scaler from resume state..')
            #     scaler.load_state_dict(resume_checkpoint_dict['scaler'])

            current_epoch = 0 #resume_checkpoint_dict['epoch']  #0 #

            return model, optimizer, lr_scheduler, scaler, current_epoch
    else:
        if cfg.TEST.TEST_MODEL_PATH or resume:
            if resume:
                logging.info('load test model from {}'.format(resume))
                msg = model.load_state_dict(torch.load(resume)['state_dict'], strict=False)  #
            else:
                logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
                msg = model.load_state_dict(torch.load(cfg.TEST.TEST_MODEL_PATH)['state_dict'], strict=False)  #

            logging.info(msg)
            print(msg)

    return model, optimizer, lr_scheduler, scaler, 0



def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
    logging.info('moving average for swa model')

def update_ema(net1, net2, iter, alpha=0.999):   ##net1:ema_model   net2:base_model

    alpha_teacher = min(1 - 1 / (iter + 1), alpha)    #1 - 1 / (iter + 1)：随着迭代越来越大
    for ema_param, param in zip(net1.parameters(), net2.parameters()):    #更新动量编码器
        if not param.data.shape:  # scalar tensor
            ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return net1

def update_sample_ema(data1, data2, iter, alpha=0.999):   ##net1:ema_model   net2:base_model

    alpha_teacher = 0.9  #min(1 - 1 / (iter + 1), alpha)    #1 - 1 / (iter + 1)：随着迭代越来越大
    if data1 is None:
        return data2
    else:
        data1.data = alpha_teacher * data1.data + (1 - alpha_teacher) * data2.data

    return data1

