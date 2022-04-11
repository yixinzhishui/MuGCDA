"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""
import math
import pickle
import torch
import torch.utils.data as data
import torch.distributed as dist
import numpy as np
import random

from segmentron.config import cfg

from torch.utils.data.sampler import Sampler, BatchSampler

__all__ = ['get_world_size', 'get_rank', 'synchronize', 'is_main_process',
           'all_gather', 'make_data_sampler', 'make_batch_data_sampler',
           'reduce_dict', 'reduce_loss_dict']


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()    #https://ptorch.com/docs/8/torch-distributed
    if world_size == 1:
        return
    dist.barrier()   #https://blog.csdn.net/weixin_41041772/article/details/109820870


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def make_data_sampler(dataset, shuffle, distributed, mode=0):     #数据采样：RandomSampler：SequentialSampler：https://zhuanlan.zhihu.com/p/100280685?utm_source=qq
    if distributed:
        if mode == 1:
            return ClassAwareDistributedSampler(dataset, shuffle=shuffle)
        else:
            #return DistributedSampler(dataset, shuffle=shuffle)    #多机多卡分布式训练：https://zhuanlan.zhihu.com/p/76638962?utm_source=wechat_session     #https://zhuanlan.zhihu.com/p/68717029
            return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    if mode == 1:
        sampler = ClassAwareSample(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0, drop_last=True):   #BatchSampler：https://zhuanlan.zhihu.com/p/100280685?utm_source=qq
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=drop_last)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler

class ClassAwareSample(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.file_to_label = data_source.file_to_label
        self.label_to_file = data_source.label_to_file
        self.file_to_index = data_source.file_to_index

    def __iter__(self):                                       #每过一轮, 权重重置， 重新计算各类采样权重
        print("----------------------------------------------------------------------")  #验证是否每过一轮，__iter__(self)重新调用一次
        label_to_file = [list() for _ in range(len(self.label_to_file))]
        for ii, file_list in enumerate(self.label_to_file):
            random.shuffle(file_list)       #不生成新列表，将原列表打乱
            self.label_to_file[ii] = file_list
        img_index = []
        class_filecount = dict()
        for i in range(cfg.DATASET.NUM_CLASSES):
            class_filecount[i] = 0
        cur_class_dist = np.zeros(cfg.DATASET.NUM_CLASSES)
        with open(r"/data/open_datasets/rsipac_semi_20000/train/rsipac_train_list_balance.txt", 'w') as file:
            for item in range(len(self.data_source)):
                if cur_class_dist.sum() == 0:
                    dist = cur_class_dist.copy()
                else:
                    dist = cur_class_dist / cur_class_dist.sum()

                w = 1 / np.log(1 + 1e2 + dist)  #4 *
                w = w / w.sum()
                c = np.random.choice(cfg.DATASET.NUM_CLASSES, p=w)
                while len(self.label_to_file[c]) == 0:
                    c = np.random.choice(cfg.DATASET.NUM_CLASSES, p=w)

                if class_filecount[c] >= (len(self.label_to_file[c]) ):  #- 1
                    np.random.shuffle(self.label_to_file[c])
                    class_filecount[c] = class_filecount[c] % (len(self.label_to_file[c]))  # - 1

                c_file = self.label_to_file[c][class_filecount[c]]
                img_index.append(self.file_to_index[c_file])
                class_filecount[c] = class_filecount[c] + 1
                cur_class_dist[self.file_to_label[c_file]] += 1
                file.write(c_file + '\n')
        return iter(img_index)

    def __len__(self):
        return len(self.data_source)

class ClassAwareSample_v2(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.file_to_label = data_source.file_to_label
        self.label_to_file = data_source.label_to_file
        self.file_to_index = data_source.file_to_index

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        pass

# class_filecount = dict()
# cur_class_dist = np.zeros(cfg.DATASET.NUM_CLASSES)
# def class_aware_sample_generator(file_to_label, label_to_file, file_to_index, iter_n):
#
#     for i in range(cfg.DATASET.NUM_CLASSES):
#         class_filecount[i] = 0
#
#     i = 0
#     while i < iter_n:
#         if cur_class_dist.sum() == 0:
#             dist = cur_class_dist.copy()
#         else:
#             dist = cur_class_dist / cur_class_dist.sum()
#
#         w = 1 / np.log(1 + 1e2 + dist)
#         w = w / w.sum()
#         c = np.random.choice(cfg.DATASET.NUM_CLASSES, p=w)
#
#         if class_filecount[c] > (len(label_to_file[c]) - 1):
#             np.random.shuffle(label_to_file[c])
#             class_filecount[c] = class_filecount[c] % (len(label_to_file[c]) - 1)
#
#         c_file = label_to_file[c][class_filecount[c]]
#
#         yield file_to_index[c_file]




class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))   #math.ceil：https://www.runoob.com/python/func-number-ceil.html
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()     #torch.Generator():https://blog.csdn.net/jiang_huixin/article/details/110282543
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ClassAwareDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.file_to_label = dataset.file_to_label
        self.label_to_file = dataset.label_to_file
        self.file_to_index = dataset.file_to_index

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))   #math.ceil：https://www.runoob.com/python/func-number-ceil.html
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # if self.shuffle:
        #     # deterministically shuffle based on epoch
        #     g = torch.Generator()     #torch.Generator():https://blog.csdn.net/jiang_huixin/article/details/110282543
        #     g.manual_seed(self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # else:
        #     indices = torch.arange(len(self.dataset)).tolist()

        print("----------------------------------------------------------------------")  # 验证是否每过一轮，__iter__(self)重新调用一次
        label_to_file = [list() for _ in range(len(self.label_to_file))]
        for ii, file_list in enumerate(self.label_to_file):
            random.shuffle(file_list)  # 不生成新列表，将原列表打乱
            self.label_to_file[ii] = file_list
        indices = []
        class_filecount = dict()
        for i in range(cfg.DATASET.NUM_CLASSES):
            class_filecount[i] = 0
        cur_class_dist = np.zeros(cfg.DATASET.NUM_CLASSES)
        with open(r"/data/open_datasets/GID_RGB_15_512x512_0overlap_sub/train/GID15_train_list_balance.txt",
                  'w') as file:
            for item in range(len(self.dataset)):
                if cur_class_dist.sum() == 0:
                    dist = cur_class_dist.copy()
                else:
                    dist = cur_class_dist / cur_class_dist.sum()

                w = 1 / np.log(1 + 1e2 + 4 * dist)
                w = w / w.sum()
                c = np.random.choice(cfg.DATASET.NUM_CLASSES, p=w)

                if class_filecount[c] > (len(self.label_to_file[c])):  # - 1
                    np.random.shuffle(self.label_to_file[c])
                    class_filecount[c] = class_filecount[c] % (len(self.label_to_file[c]))  # - 1

                c_file = self.label_to_file[c][class_filecount[c]]
                indices.append(self.file_to_index[c_file])
                class_filecount[c] = class_filecount[c] + 1
                cur_class_dist[self.file_to_label[c_file]] += 1
                file.write(c_file + '\n')

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch    #yield:https://www.runoob.com/w3cnote/python-yield-used-analysis.html

    def __len__(self):
        return self.num_iterations
