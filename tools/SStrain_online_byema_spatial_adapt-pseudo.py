import time
import copy
import datetime
import os
import sys
import csv

# from tensorboardX import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume, SegmentationScale, update_ema, update_sample_ema
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.losses import get_segmentation_losses
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg

from segmentron.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, one_mix)
from segmentron.utils.visualize import subplotimg

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
        train_source_dataset = get_segmentation_dataset(cfg.TRAIN.SOURCE.DATASET_NAME,
                                                        root=cfg.TRAIN.SOURCE.ROOT_PATH,
                                                        data_list_root=cfg.TRAIN.SOURCE.DATA_LIST,
                                                        split='train',
                                                        mode='train',
                                                        **data_kwargs)  # data_list_root=cfg.DATASET.DATA_LIST,
        train_target_dataset = get_segmentation_dataset(cfg.TRAIN.TARGET.DATASET_NAME,
                                                        root=cfg.TRAIN.TARGET.ROOT_PATH,
                                                        data_list_root=cfg.TRAIN.TARGET.DATA_LIST,
                                                        split='pesudo_train',
                                                        mode='train',
                                                        **data_kwargs)  # data_list_root=cfg.DATASET.DATA_LIST,

        val_dataset = get_segmentation_dataset(cfg.VAL.DATASET_NAME,
                                               root=cfg.VAL.ROOT_PATH,
                                               data_list_root=cfg.VAL.DATA_LIST,
                                               split='val',
                                               mode=cfg.DATASET.MODE,
                                               **data_kwargs)  # data_list_root=cfg.DATASET.DATA_LIST,
        self.iters_per_epoch = len(train_source_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_source_sampler = make_data_sampler(train_source_dataset, shuffle=True, distributed=args.distributed,
                                                 mode=0)
        train_batch_source_sampler = make_batch_data_sampler(train_source_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters,
                                                             drop_last=True)  # self.max_iters,

        train_target_sampler = make_data_sampler(train_target_dataset, shuffle=True, distributed=args.distributed,
                                                 mode=0)
        train_batch_target_sampler = make_batch_data_sampler(train_target_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters,
                                                             drop_last=True)

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.VAL.BATCH_SIZE, drop_last=False)

        self.train_source_loader = data.DataLoader(dataset=train_source_dataset,
                                                   batch_sampler=train_batch_source_sampler,
                                                   num_workers=cfg.DATASET.WORKERS,
                                                   pin_memory=True)

        self.train_target_loader = data.DataLoader(dataset=train_target_dataset,
                                                   batch_sampler=train_batch_target_sampler,
                                                   num_workers=cfg.DATASET.WORKERS,
                                                   pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,  # cfg.DATASET.WORKERS
                                          pin_memory=True)  # True

        # self.SummaryWriter = SummaryWriter(log_dir=cfg.VISUAL.LOG_SAVE_DIR + time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime()), comment=cfg.TRAIN.SUMMARYWRITER_COMMENT)
        os.makedirs(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log'), exist_ok=True)
        os.makedirs(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log'), exist_ok=True)
        with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log', 'train_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'w',
                      newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['Epoch', 'Iters', 'Loss', 'lr'])
        with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)), 'w',
                      newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['Epoch', 'pixAcc', 'mIoU'])

        # create network
        self.model = get_segmentation_model().to(self.device)
        self.ema_model = get_segmentation_model().to(self.device)

        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device,
                                  input_shape=[2, cfg.DATASET.NUM_CHANNELS, cfg.TRAIN.CROP_SIZE,
                                               cfg.TRAIN.CROP_SIZE])  # 模型FLOPs(浮点计算数)概念的清晰解释：https://zhuanlan.zhihu.com/p/137719986
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  # https://pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html
            self.ema_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_model)

            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        # self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
        #                                        aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
        #                                        ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM).to(self.device)
        #self.criterion_pesudo = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM, ignore_index=255).to(self.device)
        self.criterion_pesudo = get_segmentation_losses('ce_loss_weight').to(self.device)
        self.criterion_sample = get_segmentation_losses('rkd_loss').to(self.device)
        # optimizer, for model just includes encoder, decoder(head and auxlayer).

        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)
        self.scaler = GradScaler()

        if cfg.TRAIN.MODEL_SCALE > 1:
            self.model = SegmentationScale(self.model, float(cfg.TRAIN.MODEL_SCALE))
            self.ema_model = SegmentationScale(self.ema_model, float(cfg.TRAIN.MODEL_SCALE))

        # resume checkpoint if needed
        self.model, self.optimizer, self.lr_scheduler, self.scaler, self.start_epoch = load_model_resume(self.model,
                                                                                                         self.optimizer,
                                                                                                         self.lr_scheduler,
                                                                                                         self.scaler)
        self.ema_model.load_state_dict(self.model.state_dict().copy())

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
        #iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0

        train_source_loader_iter = iter(self.train_source_loader)
        train_target_loader_iter = iter(self.train_target_loader)

        outputs_sample_tea = None
        cls_thresh = np.ones(cfg.DATASET.NUM_CLASSES) * 0.9
        for iteration in range(self.start_epoch * iters_per_epoch + 1, self.max_iters + 1):
            self.optimizer.zero_grad()

            images, targets, _files = train_source_loader_iter.next()
            # print('images:{}'.format(images.shape))
            epoch = iteration // iters_per_epoch

            images = images.to(self.device)
            targets = targets.long().to(self.device)  # targets = targets.to(self.device)

            batch_size = images.shape[0]
            means = torch.as_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(self.device)
            stds = torch.as_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(self.device)

            outputs = self.model(images)
            # outputs_pesudo = self.model(images_pesudo)

            loss_dict = self.criterion(outputs, targets)
            # loss_dict_pesudo = self.criterion_pesudo(outputs_pesudo, targets_pesudo)
            losses = sum(loss for loss in loss_dict.values())  # + sum(loss for loss in loss_dict_pesudo.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            # loss_dict_reduced_pesudo = reduce_loss_dict(loss_dict_pesudo)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())  # + sum(loss for loss in loss_dict_reduced_pesudo.values())
            losses.backward()

            # outputs_sample = outputs.detach()
            # outputs_sample = torch.softmax(outputs_sample, dim=1).mean(dim=2).mean(dim=2)
            # outputs_sample_tea = update_sample_ema(outputs_sample_tea, outputs_sample, iteration)

            images_pesudo, _, _ = train_target_loader_iter.next()
            images_pesudo = images_pesudo.to(self.device)
            outputs_pesudo_ema = self.ema_model(images_pesudo)

            pesudo_softmax = torch.softmax(outputs_pesudo_ema.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(pesudo_softmax, dim=1)

            pseudo_prob_np, pseudo_label_np = pseudo_prob.data.cpu().numpy(), pseudo_label.data.cpu().numpy()
            logits_cls_dict = {c: [cls_thresh[c]] for c in range(cfg.DATASET.NUM_CLASSES)}
            for cls in range(cfg.DATASET.NUM_CLASSES):
                logits_cls_dict[cls].extend(pseudo_prob_np[pseudo_label_np == cls].astype(np.float16))

            tmp_cls_thresh = self.adaptive_slice_thresh(logits_cls_dict, 0.6, cfg.DATASET.NUM_CLASSES)
            cls_thresh = 0.9 * cls_thresh + (1 - 0.9) * tmp_cls_thresh

            # print('-------------------------------cls_thresh:{}'.format(cls_thresh))

            high_thresh, low_thresh = np.max(cls_thresh), np.min(cls_thresh)  #cls_thresh[:-1]
            # print('high_thresh:', high_thresh)
            # print('low_thresh:', low_thresh)
            ps_large_retain = pseudo_prob.ge(high_thresh).long() == 1
            ps_large_p = pseudo_prob.ge(low_thresh).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            # print("------------ps_size", ps_size)
            # pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = (torch.sum(ps_large_p).item() - torch.sum(ps_large_retain).item()) / ps_size
            # print('------------pseudo_weight:{}'.format(pseudo_weight))
            pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=self.device) # torch.ones(pseudo_prob.shape, device=self.device) #pseudo_weight * torch.ones(pseudo_prob.shape, device=self.device)
            pseudo_weight[ps_large_p != 1] = 0
            pseudo_weight[ps_large_retain == 1] = 1
            # print("------------all:", ps_size)
            # print("------------low:", torch.sum(ps_large_p).item())
            # print("------------high:", torch.sum(ps_large_retain).item())

            outputs_sample = self.ema_model(images)
            outputs_sample = outputs_sample.detach()
            outputs_sample_tea = torch.softmax(outputs_sample, dim=1).mean(dim=2).mean(dim=2)  #outputs_sample
            # outputs_sample_tea = update_sample_ema(outputs_sample_tea, outputs_sample, iteration)

            # print("------------ps_large_p", torch.sum(ps_large_p != 1))
            outputs_pesudo = self.model(images_pesudo)
            loss_pixel = self.criterion_pesudo(outputs_pesudo, pseudo_label, pseudo_weight)

            outputs_sample_stu = torch.softmax(outputs_pesudo, dim=1).mean(dim=2).mean(dim=2)
            loss_sample = self.criterion_sample(outputs_sample_stu, outputs_sample_tea)

            losses_pesudo = loss_pixel + loss_sample * 0.1
            losses_pesudo.backward()

            loss_dict_pixel_reduced = reduce_loss_dict(dict(loss=loss_pixel))
            losses_pixel_reduced = sum(loss for loss in loss_dict_pixel_reduced.values())
            loss_dict_sample_reduced = reduce_loss_dict(dict(loss=loss_sample))
            losses_sample_reduced = sum(loss for loss in loss_dict_sample_reduced.values())

            # self.optimizer.zero_grad()
            # losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.ema_model = update_ema(self.ema_model, self.model, iteration)

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss Source: {:.4f} || Loss Mix: {:.4f} || Loss Sample: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(), losses_pixel_reduced.item() * 0.9, losses_sample_reduced.item() * 0.1,
                        # losses_mix_reduced.item()
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))  # losses_reduced.item()
                with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log', 'train_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)),
                          'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch, iteration, losses_reduced.item(),
                                         self.optimizer.param_groups[0]['lr']])  # losses_reduced.item()

                # self.SummaryWriter.add_scalar("train-loss", losses_reduced.item(), (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.add_scalar("train-lr", self.optimizer.param_groups[0]['lr'], (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.add_scalars("train", {"loss": losses_reduced.item(), "lr":self.optimizer.param_groups[0]['lr']}, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.close()

                # self.SummaryWriter.close()
            if not self.args.skip_val and iteration % val_per_iters == 0:
                pixAcc, mIoU = self.validation(epoch, iteration)
                # self.SummaryWriter.add_scalar("pixAcc", pixAcc, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.add_scalar("mIoU", mIoU, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.close()
                self.model.train()
            if iteration % 100 == 0:
                out_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR,
                                       'class_mix_debug_online-ST-Spatial_debug0504')
                os.makedirs(out_dir, exist_ok=True)
                vis_img = torch.clamp(denorm(images, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(images_pesudo, means, stds), 0, 1)


                for j in range(batch_size):
                    vis_img_cv = np.rollaxis((vis_img[j] * 255).cpu().data.numpy().astype(np.uint8), 0, 3)[:, :, ::-1]
                    vis_trg_img_cv = np.rollaxis((vis_trg_img[j] * 255).cpu().data.numpy().astype(np.uint8), 0, 3)[:, :, ::-1]
                    vis_target_cv = self.decode_segmap(targets[j].cpu().data.numpy().astype(np.uint8))[:, :, ::-1]
                    vis_pseudo_cv = self.decode_segmap(pseudo_label[j].cpu().data.numpy().astype(np.uint8))[:, :, ::-1]
                    vis_weight_cv = (pseudo_weight[j] * 255).cpu().data.numpy().astype(np.uint8)

                    rows, cols = 2, 4
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(
                        axs[0][1],
                        targets[j],
                        'Source Seg GT',
                        cmap='cityscapes')
                    subplotimg(
                        axs[1][1],
                        pseudo_label[j],
                        'Target Seg (Pseudo) GT',
                        cmap='cityscapes')
                    # subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                    # subplotimg(
                    #     axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                    # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                    #            cmap="cityscapes")
                    subplotimg(
                        axs[1][2], ps_large_p[j], 'Pseudo Thrsed', cmap='cityscapes')
                    subplotimg(
                        axs[0][2], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(iteration + 1):06d}_{j}.png'))
                    plt.close()

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, iteration, mIoU, self.optimizer, self.lr_scheduler, is_best=False)

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
                    output = model(image)  # , return_auxilary=False
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    output = torch.argmax(output, 1)
                else:
                    size = image.size()[2:]
                    pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
                    pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
                    image = F.pad(image, (0, pad_height, 0, pad_width))  # 当输入尺寸较小时，先填补到patch尺寸，即裁减尺寸，再输入至网络
                    output = model(image)  # , return_auxilary=False
                    output = output[..., :size[0], :size[1]]
                    output = torch.argmax(output, 1)

            self.metric.update(output, target)
            pixAcc, mIoU, category_iou, category_pixAcc = self.metric.get(return_category_iou=True)
            logging.info(
                "[EVAL] Sample: {:d}, pixAcc: {:.3f}, FWIoU: {:.3f}, IOU: {}".format(i + 1, pixAcc * 100, mIoU * 100,
                                                                                     category_iou))


        pixAcc, mIoU = self.metric.get()
        with open(
                os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)),
                'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, pixAcc * 100, mIoU * 100])
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        synchronize()
        if self.best_pred < mIoU and self.save_to_disk:
            self.best_pred = mIoU
            logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, save the model..'.format(epoch,
                                                                                                                  pixAcc * 100,
                                                                                                                  mIoU * 100))
            save_checkpoint(model, epoch, iteration, self.best_pred, self.optimizer, self.lr_scheduler, self.scaler,
                            is_best=True)

        return pixAcc, mIoU

    def decode_segmap(self, mask):

        color_map = cfg.DATASET.CLASS_INDEX[0]
        assert len(mask.shape) == 2, "the expected length of the label shape is 2, but get {}".format(len(mask.shape))
        height, width = mask.shape
        decode_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for pixel, color in color_map.items():
            if isinstance(color, list) and len(color) == 3:
                decode_mask[np.where(mask == int(pixel))] = color
            else:
                print("unexpected format of color_map in the config json:{}".format(color))

        return decode_mask.astype(np.uint8)

    def adaptive_slice_thresh(self, conf_dict, tgt_portion, num_lcass):
        cls_thresh = np.ones(num_lcass, dtype=np.float32)
        for idx_cls in np.arange(0, num_lcass):
            if conf_dict[idx_cls] != None:
                arr = np.array(conf_dict[idx_cls])
                cls_thresh[idx_cls] = np.percentile(arr, tgt_portion * 100)
        return cls_thresh

if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)
    trainer = Trainer(args)
    trainer.train()