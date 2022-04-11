import logging
import os
import sys

__all__ = ['setup_logger']


def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    if distributed_rank > 0:
        return

    logging.root.name = name
    logging.root.setLevel(logging.INFO)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)    ##日志：https://blog.csdn.net/liuchunming033/article/details/39080457   logger的层次结构，父logger,子logger：https://www.cnblogs.com/i-honey/p/8052579.html
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.root.addHandler(fh)
