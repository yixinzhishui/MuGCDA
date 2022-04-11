import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import warnings
from einops import rearrange

# from .segbase import SegBaseModel
# from .model_zoo import MODEL_REGISTRY
from segmentron.models.model_zoo import MODEL_REGISTRY    #调试本脚本时
from segmentron.models.segbase import SegBaseModel        #调试本脚本时

@MODEL_REGISTRY.register(name='LawinFormer')       #编译原理，打断点调试
class LawinFormer(SegBaseModel):

    def __init__(self):
        super(LawinFormer, self).__init__()

        self.embedding_dim = 256

        self.in_channels = self.encoder.embed_dims

        self.decoder = LawinHead(in_channels=self.in_channels,  embed_dim=self.embedding_dim, num_classes=self.nclass)
        self.align_corners = False
        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.nclass, kernel_size=1, bias=False)

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

        # param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        # cls = self.classifier(_x4)
        output = self.decoder(_x)
        output = resize(input=output, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
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


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ModuleList([
            nn.MaxPool2d(patch_size, patch_size),
            nn.AvgPool2d(patch_size, patch_size)
        ])

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))

        x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        return x


class LawinAttn(nn.Module):
    def __init__(self, in_ch=512, head=4, patch_size=8, reduction=2) -> None:
        super().__init__()
        self.head = head

        self.position_mixing = nn.ModuleList([
            nn.Linear(patch_size * patch_size, patch_size * patch_size)
            for _ in range(self.head)])

        self.inter_channels = max(in_ch // reduction, 1)
        self.g = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch)
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, C, H, W = context.shape
        context = context.reshape(B, C, -1)
        context_mlp = []

        for i, pm in enumerate(self.position_mixing):
            context_crt = context[:, (C // self.head) * i:(C // self.head) * (i + 1), :]
            context_mlp.append(pm(context_crt))

        context_mlp = torch.cat(context_mlp, dim=1)
        context = context + context_mlp
        context = context.reshape(B, C, H, W)

        g_x = self.g(context).view(B, self.inter_channels, -1)
        g_x = rearrange(g_x, "b (h dim) n -> (b h) dim n", h=self.head)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(query).view(B, self.inter_channels, -1)
        theta_x = rearrange(theta_x, "b (h dim) n -> (b h) dim n", h=self.head)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(context).view(B, self.inter_channels, -1)
        phi_x = rearrange(phi_x, "b (h dim) n -> (b h) dim n", h=self.head)

        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)

        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, "(b h) n dim -> b n (h dim)", h=self.head)
        y = y.permute(0, 2, 1).contiguous().reshape(B, self.inter_channels, *query.shape[-2:])

        output = query + self.conv_out(y)
        return output


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class LawinHead(nn.Module):
    def __init__(self, in_channels: list, embed_dim=768, num_classes=19) -> None:
        super().__init__()
        for i, dim in enumerate(in_channels):
            self.add_module(f"linear_c{i + 1}", MLP(dim, 48 if i == 0 else embed_dim))

        self.linear_fuse = ConvModule(embed_dim * 3, embed_dim)
        # self.short_path = ConvModule(embed_dim*3, embed_dim)

        self.lawin_8 = LawinAttn(embed_dim, 64)
        self.lawin_4 = LawinAttn(embed_dim, 16)
        self.lawin_2 = LawinAttn(embed_dim, 4)
        self.ds_8 = PatchEmbed(8)
        self.ds_4 = PatchEmbed(4)
        self.ds_2 = PatchEmbed(2)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim)
        )

        self.cat = ConvModule(embed_dim * 5, embed_dim)

        self.low_level_fuse = ConvModule(embed_dim + 48, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def get_lawin_att_feats(self, x: Tensor, patch_size: int):
        _, _, H, W = x.shape
        query = F.unfold(x, patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size,
                          nh=H // patch_size, nw=W // patch_size)
        outs = []

        for r in [8, 4, 2]:
            context = F.unfold(x, patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            context = rearrange(context, "b (c ph pw) (nh nw) -> (b nh nw) c ph pw", ph=patch_size * r,
                                pw=patch_size * r, nh=H // patch_size, nw=W // patch_size)
            context = getattr(self, f"ds_{r}")(context)
            output = getattr(self, f"lawin_{r}")(query, context)
            output = rearrange(output, "(b nh nw) c ph pw -> b c (nh ph) (nw pw)", ph=patch_size, pw=patch_size,
                               nh=H // patch_size, nw=W // patch_size)
            outs.append(output)
        return outs

    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [self.linear_c2(features[1]).permute(0, 2, 1).reshape(B, -1, *features[1].shape[-2:])]

        for i, feature in enumerate(features[2:]):
            cf = eval(f"self.linear_c{i + 3}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape

        # feat_short = self.short_path(torch.cat(outs[::-1], dim=1))
        feat_short = feat
        feat_lawin = self.get_lawin_att_feats(feat, 8)

        feat_pool = F.interpolate(self.image_pool(feat), size=(H, W), mode='bilinear', align_corners=False)

        output = self.cat(torch.cat([feat_short, *feat_lawin, feat_pool], dim=1))
        output = F.interpolate(output, size=features[0].shape[-2:], mode='bilinear', align_corners=False)

        c1 = self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])

        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))

        seg = self.linear_pred(self.dropout(fused))
        return seg



if __name__ == '__main__':
    wetr = LawinFormer().to(torch.device('cuda:0'))
    dummy_input = torch.rand(2, 3, 512, 512).to(torch.device('cuda:0'))
    output = wetr(dummy_input)
    pass