"""Custom losses."""
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .lovasz_losses import lovasz_softmax
from ..data.dataloader import datasets
from ..config import cfg
import segmentron.solver.smp_losses as smp_loss

from segmentron.utils.registry import Registry

LOSS_REGISTRY = Registry("LOSS")


# __all__ = ['get_segmentation_loss']




@LOSS_REGISTRY.register(name='ce_loss')
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=255, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(
            inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0],
                                                               target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        if isinstance(preds, (list, tuple)):
            inputs = tuple(list(preds) + [target])
        else:
            inputs = tuple([preds, target])

        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


@LOSS_REGISTRY.register(name='sce_loss')
class SCE_Loss(nn.Module):
    def __init__(self, classes=cfg.DATASET.NUM_CLASSES, rce_alpha=0.1, rce_beta=1.0, ignore_index=-1):
        super(SCE_Loss, self).__init__()
        self.class_numbers = cfg.DATASET.NUM_CLASSES
        self.rce_alpha = rce_alpha
        self.rce_beta = rce_beta
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != self.ignore_index).float()
        labels[labels == self.ignore_index] = self.class_numbers
        label_one_hot = torch.nn.functional.one_hot(labels, self.class_numbers + 1).float()
        label_one_hot = torch.clamp(label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :], min=1e-4, max=1.0)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def forward(self, inputs, targets):
        loss = self.rce_alpha * self.ce_loss(inputs, targets) + self.rce_beta * self.rce(inputs,
                                                                                         targets.clone())

        return dict(loss=loss)


@LOSS_REGISTRY.register(name='imagebase_ce_loss')
class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes=cfg.DATASET.NUM_CLASSES, weight=None, batch_weights=False, size_average=True,
                 ignore_index=255,
                 norm=True, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, size_average,
                                   ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = batch_weights

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1))[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / (hist + 1))) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0))
        return dict(loss=loss)


@LOSS_REGISTRY.register(name='ICNet_loss')
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)



class PixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, max_samples=1024, max_views=11, ignore_index=0):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if x == 4]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, inputs, targets):
        feats = torch.nn.functional.softmax(inputs, dim=1)
        _, predict = torch.max(feats, 1)
        labels = targets.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss


# class CrossEntry_Contrast(nn.Module):
#     def __init__(self, **kwargs):
#         super(CrossEntry_Contrast, self).__init__()
#         self.crossEntry = MixSoftmaxCrossEntropyLoss(**kwargs)
#         self.contrast = PixelContrastLoss(**kwargs)
#
#     def forward(self, outputs, targets):
#         return self.crossEntry(outputs, targets) + self.contrast(outputs, targets)

@LOSS_REGISTRY.register(name='dice_sce_contrast_loss')
class BCE_SCE_Contrast(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(BCE_SCE_Contrast, self).__init__()
        self.ignore_index = ignore_index
        self.dice = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_index)

        self.sce = smp_loss.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_index)
        self.contrast = PixelContrastLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):

        loss = self.dice(inputs, targets) * 0.5 + self.sce(inputs, targets) * 0.5 + self.contrast(inputs, targets)*0.05   #
        return dict(loss=loss)



@LOSS_REGISTRY.register(name='ohem_ce_loss')
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=False,
                 **kwargs):  # use_weight=True  thresh=0.7,
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            # prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~valid_mask,
                                     1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        # target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


@LOSS_REGISTRY.register(name='encNet_loss')
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = cfg.MODEL.ENCNET.SE_LOSS
        self.se_weight = cfg.MODEL.ENCNET.SE_WEIGHT
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


@LOSS_REGISTRY.register(name='ohem_ce_loss_v2')
class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)

        if isinstance(preds, (list, tuple)):
            inputs = tuple(list(preds) + [target])
        else:
            inputs = tuple([preds, target])

        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


@LOSS_REGISTRY.register(name='lovasz_softmax')
class LovaszSoftmax(
    nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target, ignore=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


@LOSS_REGISTRY.register(name='focal_loss')
class FocalLoss(
    nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, aux=True, aux_weight=0.2, ignore_index=-1,
                 size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = self._base_forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _base_forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(
                2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, *inputs, **kwargs):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        preds, target = tuple(inputs)

        if isinstance(preds, (list, tuple)):
            inputs = tuple(list(preds) + [target])
        else:
            inputs = tuple([preds, target])
        return dict(loss=self._aux_forward(*inputs))


@LOSS_REGISTRY.register(name='binary_dice_loss')
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


@LOSS_REGISTRY.register(name='dice_loss')
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        num_classes = preds[0].shape[1]

        target_one_hot = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        loss = self._base_forward(preds[0], target_one_hot, valid_mask)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        preds, target = tuple(inputs)

        if isinstance(preds, (list, tuple)):
            inputs = tuple(list(preds) + [target])
        else:
            inputs = tuple([preds, target])

        return dict(loss=self._aux_forward(*inputs))


@LOSS_REGISTRY.register(name='ce_focalloss')
class CE_FocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CE_FocalLoss, self).__init__()
        self.ce_loss = MixSoftmaxCrossEntropyLoss()
        self.focal_loss = FocalLoss()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        loss_ce = self.ce_loss(preds, target)['loss']
        loss_focal = self.focal_loss(preds, target)['loss']

        return dict(loss=(0.5 * loss_ce + 0.5 * loss_focal))

@LOSS_REGISTRY.register(name='ohem_focalloss')
class Ohem_FocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(Ohem_FocalLoss, self).__init__()
        self.ce_loss = MixSoftmaxCrossEntropyLoss()
        self.ohem_loss = MixSoftmaxCrossEntropyOHEMLoss()
        self.focal_loss = FocalLoss()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        #loss_ce = self.ce_loss(preds, target)['loss']
        loss_ohem = self.ohem_loss(preds, target)['loss']
        loss_focal = self.focal_loss(preds, target)['loss']

        return dict(loss=(0.5 * loss_ohem + 0.5 * loss_focal))

@LOSS_REGISTRY.register(name='ce_focal_diceloss')
class CE_Focal_DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, **kwargs):
        super(CE_Focal_DiceLoss, self).__init__()
        self.ce_loss = MixSoftmaxCrossEntropyLoss(ignore_index=ignore_index)
        #self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        # self.dice_loss = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_index)

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        loss_ce = self.ce_loss(preds, target)['loss']
        #loss_focal = self.focal_loss(preds, target)['loss']
        loss_dice = self.dice_loss(preds, target)['loss']

        return dict(loss=0.5 * loss_ce + 0.5 * loss_dice) #dict(loss=0.5 * loss_ce + 0.3 * loss_focal + 0.2 * loss_dice)

@LOSS_REGISTRY.register(name='binary_focal_diceloss')
class BinaryDice_FocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BinaryDice_FocalLoss, self).__init__()
        self.binary_dice_loss = BinaryDiceLoss(**kwargs)
        self.focal_loss = FocalLoss(**kwargs)
        self.bce_loss = nn.BCELoss()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)


        return dict(loss=self.bce_loss(preds, target) + self.binary_dice_loss(preds, target))


@LOSS_REGISTRY.register(name='compose_loss')
class ComposeLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ComposeLoss, self).__init__()
        self.loss_one = MixSoftmaxCrossEntropyLoss(**kwargs)
        self.loss_two = DiceLoss(**kwargs)

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        if isinstance(preds, (list, tuple)):
            inputs = tuple(list(preds) + [target])
        else:
            inputs = tuple([preds, target])

        loss = 0.6 * self.loss_one(*inputs)['loss'] + 0.4 * self.loss_two(*inputs)['loss']

        return dict(loss=loss)



def get_segmentation_loss(model, use_ohem=False, **kwargs):
    loss_name = cfg.SOLVER.LOSS_NAME
    loss = LOSS_REGISTRY.get(loss_name)()

    return loss
