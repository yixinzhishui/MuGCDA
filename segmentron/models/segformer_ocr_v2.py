import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import warnings
import functools

# from .segbase import SegBaseModel
# from .model_zoo import MODEL_REGISTRY
from segmentron.models.model_zoo import MODEL_REGISTRY    #调试本脚本时
from segmentron.models.segbase import SegBaseModel        #调试本脚本时

if torch.__version__.startswith('0'):
    from segmentron.modules.sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    #BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    relu_inplace = True
    BatchNorm2d = nn.BatchNorm2d
    BN_MOMENTUM = 0.01
    BatchNorm2d_class = nn.BatchNorm2d

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context

class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)  #ALIGN_CORNERS

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

@MODEL_REGISTRY.register(name='SegFormer_OCR')       #编译原理，打断点调试
class SegFormer_OCR(SegBaseModel):

    def __init__(self):
        super(SegFormer_OCR, self).__init__()

        self.embedding_dim = 256
        self.ocr_mid_channels = 512
        self.ocr_key_channels = 256
        self.dropout = 0.05
        self.scale = 1

        self.feature_strides = [4, 8, 16, 32]

        self.in_channels = self.encoder.embed_dims

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,  embedding_dim=self.embedding_dim, num_classes=self.nclass)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.nclass, kernel_size=1, bias=False)

        # self.conv3x3_ocr = nn.Sequential(
        #     nn.Conv2d(self.embedding_dim, self.ocr_mid_channels,
        #               kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(self.ocr_mid_channels),
        #     nn.ReLU(inplace=relu_inplace),
        # )
        self.conv3x3_ocr = nn.Conv2d(self.embedding_dim, self.ocr_mid_channels, kernel_size=3, stride=1, padding=1)

        self.ocr_gather_head = SpatialGather_Module(self.nclass)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.ocr_mid_channels,
                                                 key_channels=self.ocr_key_channels,
                                                 out_channels=self.ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            self.ocr_mid_channels, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)


    def _forward_cam(self, x):

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        param_groups[2].append(self.classifier.weight)

        for param in list(self.conv3x3_ocr.parameters()):
            param_groups[2].append(param)
        for param in list(self.ocr_gather_head.parameters()):
            param_groups[2].append(param)
        for param in list(self.ocr_distri_head.parameters()):
            param_groups[2].append(param)
        for param in list(self.cls_head.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, return_auxilary=True):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x    #_x1.shape:torch.Size([2, 32, 128, 128])   _x2.shape:torch.Size([2, 64, 64, 64])   _x3.shape:torch.Size([2, 160, 32, 32])  _x4.shape:torch.Size([2, 256, 16, 16])
        # cls = self.classifier(_x4)
        out_aux, feats = self.decoder(_x)     #x_feats.shape:torch.Size([2, 256, 128, 128])

        feats = self.conv3x3_ocr(feats)  # feats.shape:torch.Size([1, 512, 128, 128])

        context = self.ocr_gather_head(feats, out_aux)  # context.shape:torch.Size([1, 512, 9, 1])
        feats = self.ocr_distri_head(feats, context)  # feats.shape:torch.Size([1, 512, 128, 128])

        out = self.cls_head(feats)  # out.shape:torch.Size([1, 9, 128, 128])
        out_aux = resize(input=out_aux, size=x.shape[2:], mode='bilinear', align_corners=True)
        out = resize(input=out, size=x.shape[2:], mode='bilinear', align_corners=True)

        out_aux_seg = []
        out_aux_seg.append(out)
        out_aux_seg.append(out_aux)

        if return_auxilary:
            return tuple(out_aux_seg)
        return (out + 0.4 * out_aux) / 1.4
        #return output







class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):     #x.shape:torch.Size([2, 256, 16, 16])
        x = x.flatten(2).transpose(1, 2)    #x.shape:torch.Size([2, 256, 256])
        x = self.proj(x)     #x.shape:torch.Size([2, 256, 256])
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

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

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
            norm_cfg=dict(type='BN', requires_grad=True)   #SyncBN
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):

        c1, c2, c3, c4 = x    #c4.shape:torch.Size([2, 256, 16, 16])  _c3.shape:torch.Size([2, 160, 32, 32])   #c1.shape:torch.Size([2, 32, 128, 128])

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])   #:_c4.shape:torch.Size([2, 256, 16, 16])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)   #_c4.shape:torch.Size([2, 256, 128, 128])

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))   #_c.shape:torch.Size([2, 256, 128, 128])

        x_feat = self.dropout(_c)
        x = self.linear_pred(x_feat)  #x.shape:torch.Size([2, 9, 128, 128])

        return x, x_feat


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