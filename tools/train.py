
import time
import copy
import datetime
import os
import sys
import csv
#from tensorboardX import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume, SegmentationScale
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg

class Trainer(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                                 root=cfg.TRAIN.ROOT_PATH,
                                                 data_list_root=cfg.TRAIN.DATA_LIST,
                                                 split='train',
                                                 mode='train',
                                                 **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                               root=cfg.VAL.ROOT_PATH,
                                               data_list_root=cfg.VAL.DATA_LIST,
                                               split='val',
                                               mode=cfg.DATASET.MODE,
                                               **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed, mode=0)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.VAL.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)    #True

        if not os.path.exists(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log')):
            os.makedirs(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log'))
        if not os.path.exists(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log')):
            os.makedirs(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log'))
        
        with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log', 'train_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Epoch', 'Iters', 'Loss', 'lr'])
        
        with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Epoch', 'pixAcc', 'mIoU'])

        # create network
        self.model = get_segmentation_model().to(self.device)

        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device, input_shape=[2, cfg.DATASET.NUM_CHANNELS, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE])    #模型FLOPs(浮点计算数)概念的清晰解释：https://zhuanlan.zhihu.com/p/137719986
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)      #https://pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html

            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        # self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
        #                                        aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
        #                                        ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                             iters_per_epoch=self.iters_per_epoch)

        self.scaler = GradScaler()

        # resume checkpoint if needed
        self.model, self.optimizer, self.lr_scheduler, self.scaler, self.start_epoch = load_model_resume(self.model, self.optimizer, self.lr_scheduler, self.scaler)

        if cfg.TRAIN.MODEL_SCALE > 1:
            self.model = SegmentationScale(self.model, float(cfg.TRAIN.MODEL_SCALE))

        if cfg.TRAIN_STEP_ADD:
            cfg.__setattr__("UTILS.EPOCH_STOP", self.start_epoch + 7)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(cfg.DATASET.NUM_CLASSES, args.distributed)
        self.best_pred = 0.0

    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)


            with autocast(enabled=True if cfg.TRAIN.AMP else False):
                outputs = self.model(images)
                #outputs = outputs.div(1.8)
                loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())


            if cfg.TRAIN.AMP:
                self.scaler.scale(losses).backward()
            else:
                losses.backward()
            if iteration % cfg.TRAIN.GRAD_STEPS == 0:
                if cfg.TRAIN.AMP:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))    #losses_reduced.item()
                with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log', 'train_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch, iteration, losses_reduced.item(), self.optimizer.param_groups[0]['lr']])   #losses_reduced.item()

                #self.SummaryWriter.add_scalar("train-loss", losses_reduced.item(), (epoch - 1) * iters_per_epoch + iteration)
                #self.SummaryWriter.add_scalar("train-lr", self.optimizer.param_groups[0]['lr'], (epoch - 1) * iters_per_epoch + iteration)
                #self.SummaryWriter.add_scalars("train", {"loss": losses_reduced.item(), "lr":self.optimizer.param_groups[0]['lr']}, (epoch - 1) * iters_per_epoch + iteration)
                #self.SummaryWriter.close()

            pixAcc, mIoU = 0, 0
                #self.SummaryWriter.close()
            if not self.args.skip_val and iteration % val_per_iters == 0 and epoch > cfg.UTILS.VAL_START:
                pixAcc, mIoU = self.validation(epoch, iteration)
                #self.SummaryWriter.add_scalar("pixAcc", pixAcc, (epoch - 1) * iters_per_epoch + iteration)
                #self.SummaryWriter.add_scalar("mIoU", mIoU, (epoch - 1) * iters_per_epoch + iteration)
                #self.SummaryWriter.close()
                self.model.train()
            if iteration % self.iters_per_epoch == 0 and self.save_to_disk and epoch > cfg.UTILS.VAL_START:
                save_checkpoint(self.model, epoch, iteration, mIoU, self.optimizer, self.lr_scheduler, self.scaler, is_best=False)

            if epoch > cfg.UTILS.EPOCH_STOP:
                break
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch, iteration):

        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    output = torch.argmax(output, 1)
                else:
                    size = image.size()[2:]
                    pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
                    pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
                    image = F.pad(image, (0, pad_height, 0, pad_width))
                    output = model(image)   #, return_auxilary=False
                    output = output[..., :size[0], :size[1]]
                    output = torch.argmax(output, 1)

            self.metric.update(output, target)
            pixAcc, mIoU, category_iou, category_pixAcc = self.metric.get(return_category_iou=True)
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, FWIoU: {:.3f}, IOU: {}".format(i + 1, pixAcc * 100, mIoU * 100, category_iou))

            # with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log.csv'), 'a') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow([epoch, pixAcc * 100, mIoU * 100])

        pixAcc, mIoU = self.metric.get()
        with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, pixAcc * 100, mIoU * 100])
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        synchronize()
        if self.best_pred < mIoU and self.save_to_disk:
            self.best_pred = mIoU
            logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, save the model..'.format(epoch, pixAcc * 100, mIoU * 100))
            save_checkpoint(model, epoch, iteration, self.best_pred, self.optimizer, self.lr_scheduler, self.scaler, is_best=True)

        return pixAcc, mIoU

if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)
    # debug = args.device
    # debug_2 = args.num_gpus
    # create a trainer and start train
    trainer = Trainer(args)
    trainer.train()
