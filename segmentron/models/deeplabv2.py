import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg

__all__ = ['DeepLabV2']

@MODEL_REGISTRY.register(name='DeepLabV2')       #编译原理，打断点调试
class DeepLabV2(SegBaseModel):   #deeplab  v1 v2 v3 v3+概要：https://zhuanlan.zhihu.com/p/68531147
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self):
        super(DeepLabV2, self).__init__()

        self.head = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], self.nclass)

    def forward(self, x, get_feat=False):
        size = x.size()[2:]
        c1, _, c3, c4 = self.encoder(x)

        x = self.head(c4)

        if get_feat:
            out_dict = {}
            out_dict['feat'] = c4
            out_dict['out'] = x

            return out_dict

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x



class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out