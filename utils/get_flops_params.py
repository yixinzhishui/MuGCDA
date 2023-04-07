import torch
from segmentron.models.model_zoo import get_segmentation_model, load_model_resume, SegmentationScale
from segmentron.config import cfg
import copy
from thop import profile

@torch.no_grad()     #https://www.cnblogs.com/douzujun/p/13364116.html
def show_flops_params(model, device, input_shape=[4, 3, 512, 512]):
    #summary(model, tuple(input_shape[1:]), device=device)
    input = torch.randn(*input_shape).to(torch.device(device))
    flops, params = profile(model, inputs=(input,), verbose=False)

    print('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
        model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))

if __name__ == '__main__':
    config_file = '/data_zs/code/rsipac/pytorchAI_segmentation_rsipac/configs/vaihingen_potsdam/potsdam2vaihingen_onlsource_segformer.yaml'
    cfg.update_from_file(config_file)

    device = torch.device('cuda:0')
    model = get_segmentation_model().to(device)
    show_flops_params(copy.deepcopy(model), device, input_shape=[4, 3, 512, 512])