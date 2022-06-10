# Obtained from: https://github.com/vikolss/DACS

import kornia
import numpy as np
import torch
import torch.nn as nn
import random

def strong_transform(data=None, target=None, mean=None, std=None):
    assert ((data is not None) or (target is not None))
    # data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=random.uniform(0, 1),
        s=0.2,
        p=0.2,
        mean=mean,
        std=std,
        data=data,
        target=target)
    data, target = gaussian_blur(blur=random.uniform(0, 1), data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


# def denorm(img, mean, std):
#     return img.mul(std).add(mean)  #img.mul(std).add(mean) / 255.0

def denorm(img, mean, std):
    return img.mul(std).add(mean)  #img.mul(std).add(mean) / 255.0

def denorm_(img, mean, std):
    img.mul_(std).add_(mean)   #.div_(255.0)


def renorm_(img, mean, std):
    img.sub_(mean).div_(std)   #.mul_(255.0)

def renorm(img, mean, std):
    return img.sub(mean).div(std)   #.mul_(255.0)

def color_jitter(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                # print('--------------------------------------------color')
                renorm_(data, mean, std)
    return data, target

def stong_augmentation(mean, std, data=None, target=None):
    custom_augment = [kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                      kornia.augmentation.RandomRotation(degrees=(-90, 90), p=0.5),
                      kornia.augmentation.RandomResizedCrop(scale=(0.7, 1.3), ratio=(0.8, 1.2)),
                      kornia.augmentation.RandomVerticalFlip(p=0.5)]
    if not (data is None):
        if data.shape[1] == 3:
            if random.uniform(0, 1) > 0.3:
                seq = nn.Sequential(*custom_augment)
                denorm_(data, mean, std)
                data = seq(data, label=target)   #https://kornia.readthedocs.io/en/latest/augmentation.container.html#augmentation-sequential
                # print('--------------------------------------------color')
                renorm_(data, mean, std)

    return data, target

def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
                # print('-------------------------------blur')
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes))  #.unsqueeze(0)
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label.unsqueeze(0),
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask, data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target



def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    a_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
    a_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))

    _, _, h, w = amp_src.size()
    b = (np.floor(min(h, w)*L)).astype(int)     # get b

    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, :, h1:h2, w1:w2] = a_trg[:, :, h1:h2, w1:w2]
    a_src = torch.fft.ifftshift(a_src, dim=(-2, -1))
    return a_src


def FDA_source_to_target(src_img, trg_img, L=0.1, mean=None, std=None):
    # exchange magnitude
    # input: src_img, trg_img

    L = random.uniform(0, L)
    p = random.uniform(0, 1)

    if p > 0.2:
        return src_img

    src_img = torch.clamp(denorm(src_img.clone(), mean, std), 0, 1) * 255.0
    trg_img = torch.clamp(denorm(trg_img.clone(), mean, std), 0, 1) * 255.0
    # get fft of both source and target
    fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))   #UserWarning: The function torch.rfft is deprecated and will be removed in a future PyTorch release. Use the new torch.fft module functions, instead, by importing torch.fft and calling torch.fft.fft or torch.fft.rfft.
    fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))   #UserWarning: The function torch.irfft is deprecated and will be removed in a future PyTorch release. Use the new torch.fft module functions, instead, by importing torch.fft and calling torch.fft.ifft or torch.fft.irfft

    # extract amplitude and phase of both ffts
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = amp_src_ * torch.exp(1j * pha_src)

    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1))
    # get the recomposed image: source content, target style
    src_in_trg = torch.real(src_in_trg)

    src_in_trg =torch.clip(src_in_trg, 0, 255) #torch.tensor(, dtype=torch.uint8)

    src_in_trg = renorm(src_in_trg / 255.0, mean, std)

    return src_in_trg

