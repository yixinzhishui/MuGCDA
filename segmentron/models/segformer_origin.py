import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import warnings

# from .segbase import SegBaseModel
# from .model_zoo import MODEL_REGISTRY
from segmentron.models.model_zoo import MODEL_REGISTRY    #调试本脚本时
from segmentron.models.segbase import SegBaseModel        #调试本脚本时

@MODEL_REGISTRY.register(name='SegFormer')       #编译原理，打断点调试
class SegFormer(SegBaseModel):

    def __init__(self):
        super(SegFormer, self).__init__()

        self.embedding_dim = 256
        self.feature_strides = [4, 8, 16, 32]

        self.in_channels = self.encoder.embed_dims

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,  embedding_dim=self.embedding_dim, num_classes=self.nclass)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.nclass, kernel_size=1, bias=False)

    def _forward_cam(self, x):

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)
        output = self.decoder(_x)
        output = resize(input=output, size=x.shape[2:], mode='bilinear', align_corners=True)
        return output







class MLP(nn.Module):    #MLP:类似于一个卷积层，，保持W，H不变，改变通道层的数量， 不过是通过全连接层实现，，
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):   #x.shape:torch.Size([2, 512, 16, 16])
        x = x.flatten(2).transpose(1, 2)   #x.shape:torch.Size([2, 256, 512])
        x = self.proj(x)       #x.shape::torch.Size([2, 256, 256])
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels   #64, 128, 320, 512

        #decoder_params = kwargs['decoder_params']
        #embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)   #SyncBN   BN
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):

        c1, c2, c3, c4 = x  #c1.shape:torch.Size([2, 64, 128, 128])  c2.shape:torch.Size([2, 128, 64, 64])  c3.shape:torch.Size([2, 320, 32, 32])  c4.shape:torch.Size([2, 512, 16, 16])

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])   #_c4.shape:torch.Size([2, 256, 16, 16])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])   ##_c3.shape:torch.Size([2, 256, 32, 32])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])  ##_c2.shape:torch.Size([2, 256, 64, 64])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])  #_c1.shape:torch.Size([2, 256, 128, 128])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


if __name__ == '__main__':
    wetr = SegFormer().to(torch.device('cuda:0'))
    dummy_input = torch.rand(2, 3, 512, 512).to(torch.device('cuda:0'))
    output = wetr(dummy_input)
    pass