"""Filesystem utility functions."""
from __future__ import absolute_import
import os
import errno
import torch
import logging
import shutil

from ..config import cfg

def save_checkpoint(model, epoch, optimizer=None, lr_scheduler=None, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.VISUAL.MODEL_SAVE_DIR)    #os.path.expanduser：https://segmentfault.com/a/1190000017485286
    # directory = os.path.join(directory, '{}_{}_{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE,
    #                                                          cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}.pth'.format(str(epoch))
    filename = os.path.join(directory, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        torch.save(model_state_dict, best_filename)
    else:
        save_state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        if not os.path.exists(filename):
            torch.save(save_state, filename)
            logging.info('Epoch {} model saved in: {}'.format(epoch, filename))

        # remove last epoch
        pre_filename = '{}.pth'.format(str(epoch - 1))
        pre_filename = os.path.join(directory, pre_filename)
        try:
            if os.path.exists(pre_filename):
                os.remove(pre_filename)
        except OSError as e:
            logging.info(e)

def save_checkpoint(model, current_epoch, current_iteration, eva_metric, optimizer=None, lr_scheduler=None, scaler=None, is_best=False):
    directory = os.path.expanduser(cfg.VISUAL.MODEL_SAVE_DIR)  #cfg.TRAIN.MODEL_SAVE_DIR # os.path.expanduser：https://segmentfault.com/a/1190000017485286

    # directory = os.path.join(directory, '{}_{}_{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE,
    #                                                          cfg.DATASET.NAME, cfg.TIME_STAMP))

    directory = os.path.join(directory, '{}'.format(cfg.VISUAL.CURRENT_NAME))
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    state = {
        'epoch': current_epoch,
        'iteration': current_iteration,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else optimizer,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else lr_scheduler,
        'scaler': scaler.state_dict() if scaler is not None else scaler
    }

    Current_ModelSave = os.listdir(directory)
    Current_ModelSave.sort(key=lambda x: -1 if (x.split('_'))[0] == 'best' else float((x.split('_'))[1]))   #, reverse=True   #https://www.cnblogs.com/chester-cs/p/12252358.html
    if len(Current_ModelSave) > 40:  #9
        for remove_ModelFile in Current_ModelSave[1:-40]:  #9
            os.remove(os.path.join(directory, remove_ModelFile))
    torch.save(state, os.path.join(directory, '{}_{:.5f}_checkpoint.pth.tar'.format(current_epoch, eva_metric)))
    #torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'checkpoint_{}.pth'.format(self.current_epoch)))
    if is_best:
        shutil.copyfile(os.path.join(directory, '{}_{:.5f}_checkpoint.pth.tar'.format(current_epoch, eva_metric)), os.path.join(directory, 'best_checkpoint.pth.tar'))

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

