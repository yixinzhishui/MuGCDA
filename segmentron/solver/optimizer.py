import logging
import torch.nn as nn

from torch import optim
from segmentron.config import cfg
import torch

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas,weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter:

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

def _set_batch_norm_attr(named_modules, attr, value):
    for m in named_modules:
        if isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
            setattr(m[1], attr, value)


def _get_paramters(model):
    params_list = list()
    # if hasattr(model, 'encoder') and model.encoder is not None and hasattr(model, 'decoder'):
    #     params_list.append({'params': model.encoder.parameters(), 'lr': cfg.SOLVER.LR})
    #     if cfg.MODEL.BN_EPS_FOR_ENCODER:
    #         logging.info('Set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
    #         _set_batch_norm_attr(model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)   #named_modules()：https://blog.csdn.net/watermelon1123/article/details/98036360
    #
    #     for module in model.decoder:
    #         params_list.append({'params': getattr(model, module).parameters(),
    #                             'lr': cfg.SOLVER.LR * cfg.SOLVER.DECODER_LR_FACTOR})
    #
    #     if cfg.MODEL.BN_EPS_FOR_DECODER:
    #         logging.info('Set bn custom eps for bn in decoder: {}'.format(cfg.MODEL.BN_EPS_FOR_DECODER))
    #         for module in model.decoder:
    #             _set_batch_norm_attr(getattr(model, module).named_modules(), 'eps',
    #                                      cfg.MODEL.BN_EPS_FOR_DECODER)
    # else:
    #     logging.info('Model do not have encoder or decoder, params list was from model.parameters(), '
    #                  'and arguments BN_EPS_FOR_ENCODER, BN_EPS_FOR_DECODER, DECODER_LR_FACTOR not used!')
    params_list = model.parameters()

    if cfg.MODEL.BN_MOMENTUM and cfg.MODEL.BN_TYPE in ['BN']:
        logging.info('Set bn custom momentum: {}'.format(cfg.MODEL.BN_MOMENTUM))
        _set_batch_norm_attr(model.named_modules(), 'momentum', cfg.MODEL.BN_MOMENTUM)
    elif cfg.MODEL.BN_MOMENTUM and cfg.MODEL.BN_TYPE not in ['BN']:
        logging.info('Batch norm type is {}, custom bn momentum is not effective!'.format(cfg.MODEL.BN_TYPE))

    return params_list


def get_optimizer(model):
    parameters = _get_paramters(model)
    opt_lower = cfg.SOLVER.OPTIMIZER.lower()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=cfg.SOLVER.LR, alpha=0.9, eps=cfg.SOLVER.EPSILON,
            momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'poly_adamw':
        param_groups = model.get_param_groups()
        optimizer = optim.AdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": cfg.SOLVER.LR,
                    "weight_decay": cfg.SOLVER.LR,
                },
                {
                    "params": param_groups[1],
                    "lr": cfg.SOLVER.LR,
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": cfg.SOLVER.LR * 10,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                }
            ],
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

    else:
        raise ValueError("Expected optimizer method in [sgd, adam, adadelta, rmsprop], but received "
                         "{}".format(opt_lower))

    return optimizer

"""用于train_segformer.py调试时"""
# def get_optimizer(model, max_iters, iters_per_epoch):
#     parameters = _get_paramters(model)
#     opt_lower = cfg.SOLVER.OPTIMIZER.lower()
#
#     if opt_lower == 'sgd':
#         optimizer = optim.SGD(
#             parameters, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     elif opt_lower == 'adam':
#         optimizer = optim.Adam(
#             parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     elif opt_lower == 'adadelta':
#         optimizer = optim.Adadelta(
#             parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     elif opt_lower == 'rmsprop':
#         optimizer = optim.RMSprop(
#             parameters, lr=cfg.SOLVER.LR, alpha=0.9, eps=cfg.SOLVER.EPSILON,
#             momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     elif opt_lower == 'poly_adamw':
#         param_groups = model.get_param_groups()
#         optimizer = PolyWarmupAdamW(
#             params=[
#                 {
#                     "params": param_groups[0],
#                     "lr": cfg.SOLVER.LR,
#                     "weight_decay": cfg.SOLVER.LR,
#                 },
#                 {
#                     "params": param_groups[1],
#                     "lr": cfg.SOLVER.LR,
#                     "weight_decay": 0.0,
#                 },
#                 {
#                     "params": param_groups[2],
#                     "lr": cfg.SOLVER.LR * 10,
#                     "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
#                 },
#             ],
#             lr=cfg.SOLVER.LR,
#             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             betas=[0.9, 0.999],   #cfg.optimizer.betas
#             warmup_iter=cfg.SOLVER.WARMUP.EPOCHS * iters_per_epoch,
#             max_iter=max_iters,
#             warmup_ratio=1e-6,   #cfg.scheduler.warmup_ratio
#             power=cfg.SOLVER.POLY.POWER
#         )
#     else:
#         raise ValueError("Expected optimizer method in [sgd, adam, adadelta, rmsprop], but received "
#                          "{}".format(opt_lower))
#
#     return optimizer
