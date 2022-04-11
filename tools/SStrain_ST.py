# 2020年了，语义分割还有哪些方向可以研究：https://mp.weixin.qq.com/s?__biz=MjM5MjgwNzcxOA==&mid=2247483875&idx=1&sn=3f378bffca754486d7552147898b3763&chksm=a6a1efaa91d666bc77c529286692634ef73d4340e7e6cc4609dcd056ea92777c6eda69a44e51&token=1209743408&lang=zh_CN#rd
# 弱监督语义分割综述：https://mp.weixin.qq.com/s?__biz=MjM5MjgwNzcxOA==&mid=2247483709&idx=1&sn=b7ff5fd12945d0626370abec6821bb3c&chksm=a6a1ef7491d66662c6b0f7e1783e77924f4c85941f7f1452d43649c6bd63ad0fe7b766227b4b&scene=21#wechat_redirect
import time
import copy
import datetime
import os
import sys
import csv
# from tensorboardX import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))  # https://www.cnblogs.com/joldy/p/6144813.html
root_path = os.path.split(cur_path)[0]
sys.path.append( root_path)  # sys.path.append:https://blog.csdn.net/zxyhhjs2017/article/details/80582246?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242

import logging
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch.cuda.amp import autocast   #https://zhuanlan.zhihu.com/p/165152789
from torch.cuda.amp import GradScaler
from matplotlib import pyplot as plt

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume, SegmentationScale
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
            # 数据归一化的原因：https://blog.csdn.net/qq_38765642/article/details/109779370  归一化与反归一化：https://blog.csdn.net/qq_38929105/article/details/106733564?utm_medium=distribute.pc_relevant.none-task-blog-searchFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-searchFromBaidu-1.control
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
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)  # https://pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html
            """
             同步批处理标准PyTorch
             PyTorch中的同步批处理规范化实现。
             此模块与内置的PyTorch BatchNorm不同，因为在训练过程中所有设备的均值和标准差都减小了。
             例如，当在训练期间使用nn.DataParallel封装网络时，PyTorch的实现仅使用该设备上的统计信息对每个设备上的张量进行归一化，这加快了计算速度，并且易于实现，但统计信息可能不准确。 相反，在此同步版本中，将对分布在多个设备上的所有训练样本进行统计。
             请注意，对于单GPU或仅CPU的情况，此模块的行为与内置的PyTorch实现完全相同。
             该模块目前仅是用于研究用途的原型版本。 如下所述，它有其局限性，甚至可能会遇到一些设计问题。
            """
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

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)
        self.scaler = GradScaler()

        # resume checkpoint if needed
        self.model, self.optimizer, self.lr_scheduler, self.scaler, self.start_epoch = load_model_resume(self.model,
                                                                                                         self.optimizer,
                                                                                                         self.lr_scheduler,
                                                                                                         self.scaler)

        if cfg.TRAIN.MODEL_SCALE > 1:
            self.model = SegmentationScale(self.model, float(cfg.TRAIN.MODEL_SCALE))
            print("--------------------------model scale:{}".format(cfg.TRAIN.MODEL_SCALE))

        if cfg.TRAIN_STEP_ADD:
            cfg.__setattr__("UTILS.EPOCH_STOP", self.start_epoch + 7)
        print("--------------------------epoch stop:{}".format(cfg.UTILS.EPOCH_STOP))


        if args.distributed:  # 使用PyTorch编写分布式应用程序：https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/dist_tuto.md     #https://oldpan.me/archives/pytorch-to-use-multiple-gpus   #https://zhuanlan.zhihu.com/p/76638962?utm_source=wechat_session
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


        #iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        # self.model.train()
        train_source_loader_iter = iter(self.train_source_loader)
        train_target_loader_iter = iter(self.train_target_loader)
        self.model.train()

        for iteration in range(self.start_epoch * iters_per_epoch + 1, self.max_iters + 1):

            self.optimizer.zero_grad()

            images, targets, _ = train_source_loader_iter.next()
            # print('images:{}'.format(images.shape))
            epoch = iteration // iters_per_epoch

            images = images.to(self.device)
            targets = targets.long().to(self.device)  # targets = targets.to(self.device)  损失函数输入需为Long型
            # targets = targets.float().to(self.device)  # targets = targets.to(self.device)  单分类损失函数损失函数输入需为Float型
            batch_size = images.shape[0]
            dev = images.device
            means = torch.as_tensor([128, 128, 128]).view(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(self.device)
            stds = torch.as_tensor([128, 128, 128]).view(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(self.device)

            outputs = self.model(images)
            # outputs_pesudo = self.model(images_pesudo)

            loss_dict = self.criterion(outputs, targets)
            # loss_dict_pesudo = self.criterion_pesudo(outputs_pesudo, targets_pesudo)


            losses = sum(loss for loss in loss_dict.values())   #+ sum(loss for loss in loss_dict_pesudo.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            # loss_dict_reduced_pesudo = reduce_loss_dict(loss_dict_pesudo)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())  # + sum(loss for loss in loss_dict_reduced_pesudo.values())
            losses.backward()

            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # self.lr_scheduler.step()

            # self.model.eval()
            images_pesudo, _, _ = train_target_loader_iter.next()
            images_pesudo = images_pesudo.to(self.device)
            outputs_pesudo = self.model(images_pesudo)

            pesudo_softmax = torch.softmax(outputs_pesudo.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(pesudo_softmax, dim=1)
            ps_large_p = pseudo_prob.ge(0.9).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=self.device)
            # print('pseudo_weight:{}'.format(pseudo_weight.shape))
            # pseudo_weight[:, :15, :] = 0
            # pseudo_weight[:, -15:, :] = 0

            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=self.device)
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(targets)
            for i in range(batch_size):
                # print('mix_masks[i]:{}'.format(mix_masks[i].shape))
                # print('images[i]:{}'.format(images[i].shape))
                mixed_img[i], mixed_lbl[i] = one_mix(mix_masks[i],
                                                     torch.stack((images[i], images_pesudo[i])),
                                                     torch.stack((targets[i], pseudo_label[i])))
                # print('gt_pixel_weight[i]:{}'.format(gt_pixel_weight[i].shape))
                # print('pseudo_weight[i]:{}'.format(pseudo_weight[i].shape))
                _, pseudo_weight[i] = one_mix(mix_masks[i],
                                              target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img, dim=0)
            # print('mixed_img:{}'.format(mixed_img.shape))
            mixed_lbl = torch.cat(mixed_lbl, dim=0)
            # print('mixed_lbl:{}'.format(mixed_lbl.shape))

            # self.model.train()
            outputs_mix = self.model(mixed_img)
            # print('mixed_img:{}'.format(mixed_img.shape))
            # print('mixed_lbl:{}'.format(mixed_lbl.shape))
            # print('pseudo_weight:{}'.format(pseudo_weight.shape))
            losses_mix = self.criterion_pesudo(outputs_mix, mixed_lbl, pseudo_weight)
            losses_mix.backward()

            loss_dict_mix_reduced = reduce_loss_dict(dict(loss=losses_mix))
            losses_mix_reduced = sum(loss for loss in loss_dict_mix_reduced.values())


            self.optimizer.step()
            # self.optimizer.zero_grad()
            self.lr_scheduler.step()

            # if cfg.TRAIN.AMP:
            #     self.scaler.scale(losses).backward()
            # else:
            #     losses.backward()
            # if iteration % cfg.TRAIN.GRAD_STEPS == 0:
            #     if cfg.TRAIN.AMP:
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #     else:
            #         self.optimizer.step()
            #     self.optimizer.zero_grad()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss Source: {:.4f} || Loss Mix: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(), losses_mix_reduced.item(),  #losses_mix_reduced.item()
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))  # losses_reduced.item()
                with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'train_log', 'train_log_{}.csv'.format(cfg.VISUAL.CURRENT_NAME)),
                          'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch, iteration, losses_reduced.item(),
                                         self.optimizer.param_groups[0]['lr']])  # losses_reduced.item()

                # self.SummaryWriter.add_scalar("train-loss", losses_reduced.item(), (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.add_scalar("train-lr", self.optimizer.param_groups[0]['lr'], (epoch - 1) * iters_per_epoch + iteration)    #SummaryWriter封装使用：https://www.cnblogs.com/chengebigdata/p/10121109.html    #tensorboard 平滑损失曲线代码:https://blog.csdn.net/charel_chen/article/details/80364841
                # self.SummaryWriter.add_scalars("train", {"loss": losses_reduced.item(), "lr":self.optimizer.param_groups[0]['lr']}, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.close()          #self.SummaryWriter.close()：tensorboard生成文件大小0KB：https://blog.csdn.net/york1996/article/details/103898325    #动态查看：https://blog.csdn.net/weixin_38709804/article/details/103922830

            pixAcc, mIoU = 0, 0
                # self.SummaryWriter.close()
            if not self.args.skip_val and iteration % val_per_iters == 0:
                pixAcc, mIoU = self.validation(epoch, iteration)
                self.model.train()
                # out_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR,
                #                        'class_mix_debug')
                # os.makedirs(out_dir, exist_ok=True)
                # vis_img = torch.clamp(denorm(images, means, stds), 0, 1)
                # vis_trg_img = torch.clamp(denorm(images_pesudo, means, stds), 0, 1)
                # vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
                # for j in range(batch_size):
                #     rows, cols = 2, 5
                #     fig, axs = plt.subplots(
                #         rows,
                #         cols,
                #         figsize=(3 * cols, 3 * rows),
                #         gridspec_kw={
                #             'hspace': 0.1,
                #             'wspace': 0,
                #             'top': 0.95,
                #             'bottom': 0,
                #             'right': 1,
                #             'left': 0
                #         },
                #     )
                #     subplotimg(axs[0][0], vis_img[j], 'Source Image')
                #     subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                #     subplotimg(
                #         axs[0][1],
                #         targets[j],
                #         'Source Seg GT',
                #         cmap='cityscapes')
                #     subplotimg(
                #         axs[1][1],
                #         pseudo_label[j],
                #         'Target Seg (Pseudo) GT',
                #         cmap='cityscapes')
                #     subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                #     subplotimg(
                #         axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                #     # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #     #            cmap="cityscapes")
                #     subplotimg(
                #         axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                #     subplotimg(
                #         axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                #     for ax in axs.flat:
                #         ax.axis('off')
                #     plt.savefig(
                #         os.path.join(out_dir,
                #                      f'{(iteration + 1):06d}_{j}.png'))
                #     plt.close()


                # self.SummaryWriter.add_scalar("pixAcc", pixAcc, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.add_scalar("mIoU", mIoU, (epoch - 1) * iters_per_epoch + iteration)
                # self.SummaryWriter.close()
                # self.model.train()
                # pixAcc, mIoU = self.validation(epoch, iteration)
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

            # with open(os.path.join(cfg.VISUAL.LOG_SAVE_DIR, 'valid_log', 'valid_log.csv'), 'a') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow([epoch, pixAcc * 100, mIoU * 100])

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

# ssh远程pycharm配置：https://blog.csdn.net/hesongzefairy/article/details/96276263
# https://blog.csdn.net/Alina_M/article/details/105901297?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 tools/train.py   #https://blog.csdn.net/andrew80/article/details/89189544
