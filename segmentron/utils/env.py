# this code heavily based on detectron2

import logging
import numpy as np
import os
import random
from datetime import datetime
import torch

__all__ = ["seed_all_rng"]


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()        #https://blog.csdn.net/qq_38839677/article/details/80671579
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )                     #os.urandom:https://blog.csdn.net/liuxingyu_21/article/details/18218063
        logger = logging.getLogger(__name__)        #日志：https://blog.csdn.net/liuchunming033/article/details/39080457   logger的层次结构，父logger,子logger：https://www.cnblogs.com/i-honey/p/8052579.html
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def set_random_seed(seed, deterministic=True):   #https://blog.csdn.net/yyywxk/article/details/121606566  https://blog.csdn.net/hyk_1996/article/details/84307108
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)   #
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   #cpu
    torch.cuda.manual_seed(seed)   #gpu
    torch.cuda.manual_seed_all(seed)   #all gpus
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
