import os
import logging
import json
import torch

from .distributed import get_rank, synchronize
from .logger import setup_logger
from .env import seed_all_rng, set_random_seed
from ..config import cfg

def default_setup(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if not args.no_cuda and torch.cuda.is_available():
        # cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True   #https://blog.csdn.net/weixin_34910922/article/details/107947125?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.control
        args.device = "cuda:{}".format(cfg.CURRENT_GPU)
        # args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO
    # if args.save_pred:
    #     outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)

    save_dir = cfg.VISUAL.LOG_SAVE_DIR if cfg.PHASE == 'train' else None
    setup_logger("Segmentron", save_dir, get_rank(), filename='{}.txt'.format(cfg.VISUAL.CURRENT_NAME))

    logging.info("Set current gpu device:{}".format(cfg.CURRENT_GPU))
    logging.info("Using {} GPUs".format(num_gpus))    ##日志：https://blog.csdn.net/liuchunming033/article/details/39080457   logger的层次结构，父logger,子logger：https://www.cnblogs.com/i-honey/p/8052579.html
    logging.info(args)
    logging.info(json.dumps(cfg, indent=8))

    # seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())
    set_random_seed(cfg.SEED)