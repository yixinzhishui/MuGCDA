import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('--config-file', default= r'/data_zs/code/loveDA/pytorchAI_segmentation_loveda/configs/tmp/loveda_vaihingen2potsdam_segformer.yaml', metavar="FILE", #/data_zs/code/loveDA/pytorchAI_segmentation_loveda/configs/vaihingen_potsdam/loveda_potsdam2vaihingen_segformer.yaml #/code/python/pytorch/sandong/pytorchAI_segmentation_sandong/configs/sandong_segformer.yaml    #/data_zs/code/sandong/pytorchAI_segmentation_sandong/configs/sandong_segformer.yaml
                        help='config file path')   #, required=True  #metavar：https://www.cnblogs.com/Allen-rg/p/12234237.html  自改：加default= r'configs/cityscapes_deeplabv3_plus.yaml'    #r'/data/Landcover/SegmenTron-master/configs/landcover9num_deeplabv3_plus.yaml'
    # cuda setting       #/code/python/pytorch/pytorchAI_segmentation_rsipac_transfer_v2/configs/rsipac_segformer.yaml  /workspace/configs/rsipac_segformer_test.yaml

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,                   #None  r'E:\code\python\pytorch\my_code\Multi_models\SegmenTron-master\SegmenTron-master\tools\runs\checkpoints\DeepLabV3_Plus_xception65_naic_2021-01-04-13-47\400.pth'
                        help='put the path to resuming file if needed')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # for visual
    parser.add_argument('--input-img', type=str, default=r'J:\Expert_Datasets\CloudShadow_GF1_3band_512x512_3numclass\val\images',
                        help='path to the input image or a directory of images')
    # config options
    parser.add_argument('opts', help='See config for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args