"""Model Zoo"""
from .model_zoo import MODEL_REGISTRY
from .fast_scnn import FastSCNN
from .deeplabv3_plus import DeepLabV3Plus
from .hrnet_seg import HighResolutionNet
from .fcn import FCN
from .dfanet import DFANet
from .pspnet import PSPNet
from .icnet import ICNet
from .danet import DANet
# from .ccnet import CCNet
from .bisenet import BiSeNet
from .cgnet import CGNet
from .denseaspp import DenseASPP
from .dunet import DUNet
from .encnet import EncNet
from .lednet import LEDNet
from .ocnet import OCNet
from .hardnet import HardNet
from .refinenet import RefineNet
from .dabnet import DABNet
from .unet import UNet
from .fpenet import FPENet
from .contextnet import ContextNet
from .espnetv2 import ESPNetV2
from .enet import ENet
from .edanet import EDANet
from .pointrend import PointRend
'''当加入新的模型时， 需在此进行模型导入，否则无法完成注册过程(MODEL_REGISTRY.register)'''   #https://www.cnblogs.com/tp1226/p/8453854.html
from .hrnet_ocr_seg import HighResolutionNet
from .segformer import SegFormer
# from .segformer_ocr import SegFormer_OCR
from .segformer_ocr_v2 import SegFormer_OCR
from .daformer import DAFormer
from .lawin_transformer import LawinFormer
from .deeplabv2 import DeepLabV2






