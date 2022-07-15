from .config import SegmentronConfig

cfg = SegmentronConfig()

########################## basic set ###########################################
# random seed
cfg.SEED = 1024
# train time stamp, auto generate, do not need to set
cfg.TIME_STAMP = ''
# root path
#cfg.ROOT_PATH = ''
# model phase ['train', 'test']
cfg.PHASE = 'train'    #'train'
cfg.CURRENT_GPU = 0
cfg.TRAIN_STEP_ADD = False
########################## dataset config #########################################
# dataset name
cfg.DATASET.NAME = ''
# pixel mean
cfg.DATASET.MEAN = [0.33403504, 0.37003833, 0.32259592] #[0.5, 0.5, 0.5]
# pixel std
cfg.DATASET.STD = [0.14451134, 0.12238872, 0.11366305] #[0.5, 0.5, 0.5]
cfg.DATASET.ROOT_PATH = ''
cfg.DATASET.NUM_CHANNELS = 3
# dataset ignore index
cfg.DATASET.IGNORE_INDEX = -1
# workers
cfg.DATASET.WORKERS = 4 #32 #32
# val dataset mode
cfg.DATASET.MODE = 'val' #'testval'
cfg.DATASET.NUM_CLASSES = 16
cfg.DATASET.CLASS_INDEX = [[200, 0, 0],
                  [250, 0, 150],
                  [200, 150, 150],
                  [250, 150, 150],
                  [0, 200, 0],
                  [150, 250, 0],
                  [150, 200, 150],
                  [200, 0, 200],
                  [150, 0, 250],
                  [150, 150, 250],
                  [250, 200, 0],
                  [200, 200, 0],
                  [0,   0, 200],
                  [0,   150, 200],
                  [0,   200, 250]]   #[0, 255]#[0, 9, 20, 30, 40, 50, 60, 90, 255]
########################### data augment ######################################
# data augment image mirror
cfg.AUG.MIRROR = True
# blur probability
cfg.AUG.BLUR_PROB = 0.0
# blur radius
cfg.AUG.BLUR_RADIUS = 0.0
# color jitter, float or tuple: (0.1, 0.2, 0.3, 0.4)
cfg.AUG.COLOR_JITTER = None
########################### train config ##########################################
# epochs
cfg.TRAIN.EPOCHS = 30
# batch size
cfg.TRAIN.BATCH_SIZE = 1
# train crop size
cfg.TRAIN.GRAD_STEPS = 2

cfg.TRAIN.MODEL_SCALE = 1

cfg.TRAIN.CROP_SIZE = 512  #769
# train base size
cfg.TRAIN.BASE_SIZE = 512  #1024
# model output dir
cfg.TRAIN.ROOT_PATH = ''
cfg.TRAIN.DATA_LIST = None


#uda dataset
cfg.TRAIN.SOURCE.DATASET_NAME = None
cfg.TRAIN.SOURCE.ROOT_PATH = None
cfg.TRAIN.SOURCE.DATA_LIST = None
cfg.TRAIN.TARGET.DATASET_NAME = None
cfg.TRAIN.TARGET.ROOT_PATH = None
cfg.TRAIN.TARGET.DATA_LIST = None

cfg.TRAIN.FOLD = 0
cfg.TRAIN.DATA_AUGMENT = 'train_transform'

cfg.TRAIN.SUMMARYWRITER_COMMENT = 'deeplab_v3+'
# pretrained model for eval or finetune
cfg.TRAIN.PRETRAINED_MODEL_PATH = None #r''
# use pretrained backbone model over imagenet
cfg.TRAIN.BACKBONE_PRETRAINED = True
# backbone pretrained model path, if not specific, will load from url when backbone pretrained enabled    tf-xception65-270e81cf.pth
cfg.TRAIN.BACKBONE_PRETRAINED_PATH = r'/data_zs/data/pretrained_models/mit_b1.pth' # r'/data/Landcover/pretrain_pth/tf-xception65-270e81cf.pth' #r'E:\code\python\pytorch\my_code\Multi_models\SegmenTron-master\pretrain_pth\tf-xception65-270e81cf.pth'  #r'/data/CloudShadow/pretrain_pth/tf-xception65-270e81cf.pth'    #r'E:\code\python\pytorch\github_code\Multi_models\SegmenTron-master\pretrain_pth\deeplabv3_plus_xception_segmentron.pth'
# resume model path                 #r'E:\code\python\pytorch\my_code\Multi_models\SegmenTron-master\pretrain_pth\tf-xception65-270e81cf.pth'   r'/data/NAIC/pretrain_pth/deeplabv3_plus_xception_segmentron.pth'
#cfg.TRAIN.RESUME_MODEL_PATH = ''
# whether to use synchronize bn
cfg.TRAIN.SYNC_BATCH_NORM = True
# save model every checkpoint-epoch
cfg.TRAIN.SNAPSHOT_EPOCH = 10
cfg.TRAIN.AMP = False


########################## val config ###########################################
cfg.VAL.DATASET_NAME = None
cfg.VAL.BATCH_SIZE = 1
cfg.VAL.CROP_SIZE = 512
cfg.VAL.ROOT_PATH = ''
cfg.VAL.DATA_LIST = None
########################## test config ###########################################
# val/test model path
cfg.TEST.TEST_MODEL_PATH = r'' #r'E:\data\projects_and_topic\CloudShadow_detection\runs\checkpoints\DeepLabV3_Plus_xception65_cloudshadow_2021-01-29-13-12\best_checkpoint.pth.tar'    #E:\code\python\pytorch\my_code\Multi_models\SegmenTron-master\SegmenTron-master\tools\runs\checkpoints\DeepLabV3_Plus_xception65_cloudshadow_2021-01-29-10-42\29_checkpoint.pth.tar
# test batch size
cfg.TEST.BATCH_SIZE = 1
# eval crop size
cfg.TEST.CROP_SIZE = None   #None   512
# multiscale eval
cfg.TEST.SCALES = [1.0]
# flip
cfg.TEST.FLIP = False
cfg.TEST.USE_TTA = False

########################### optimizer config ##################################
# base learning rate
cfg.SOLVER.LR = 1e-4
# optimizer method
cfg.SOLVER.OPTIMIZER = "sgd"  #"sgd"  #"adam"
# optimizer epsilon
cfg.SOLVER.EPSILON = 1e-8
# optimizer momentum
cfg.SOLVER.MOMENTUM = 0.9
# weight decay
cfg.SOLVER.WEIGHT_DECAY = 0.01 #0.01 #1e-4 #0.00004
# decoder lr x10
cfg.SOLVER.DECODER_LR_FACTOR = 10.0
# lr scheduler mode
cfg.SOLVER.LR_SCHEDULER = "poly"    #poly   step

cfg.SOLVER.COSINEANNEALINGLR.T_MAX = 8
cfg.SOLVER.COSINEANNEALINGLR.T_MULT = 1
# poly power
cfg.SOLVER.POLY.POWER = 2 #4
# step gamma
cfg.SOLVER.STEP.GAMMA = 0.2
# milestone of step lr scheduler
cfg.SOLVER.STEP.DECAY_EPOCH = [10, 20, 30, 50]
# warm up epochs can be float
cfg.SOLVER.WARMUP.EPOCHS = 0 #8 #4 #0 #1.  #3
# warm up factor
cfg.SOLVER.WARMUP.FACTOR = 1.0 / 3
# warm up method
cfg.SOLVER.WARMUP.METHOD = 'linear'
# whether to use ohem
cfg.SOLVER.OHEM = False
# whether to use aux loss
cfg.SOLVER.AUX = False
# aux loss weight
cfg.SOLVER.AUX_WEIGHT = 0.4
# loss name
cfg.SOLVER.LOSS_NAME = ''

########################## model #######################################
cfg.MODEL.SOURCE = ''
# model name
cfg.MODEL.MODEL_NAME = ''
# model backbone
cfg.MODEL.BACKBONE = 'mit_b1' #''
cfg.MODEL.ACTIVATION = ''
cfg.MODEL.THRESHOLD = 0.0
# model backbone channel scale
cfg.MODEL.BACKBONE_SCALE = 1.0
# support resnet b, c. b is standard resnet in pytorch official repo
# cfg.MODEL.RESNET_VARIANT = 'b'
# multi branch loss weight
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0]
# gn groups
cfg.MODEL.DEFAULT_GROUP_NUMBER = 16
# whole model default epsilon
cfg.MODEL.DEFAULT_EPSILON = 1e-5
# batch norm, support ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
cfg.MODEL.BN_TYPE = 'BN'
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_ENCODER = None
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_DECODER = None
# backbone output stride
cfg.MODEL.OUTPUT_STRIDE = 16
# BatchNorm momentum, if set None will use api default value.
cfg.MODEL.BN_MOMENTUM = None

########################## DANet config ####################################
# danet param
cfg.MODEL.DANET.MULTI_DILATION = None
# danet param
cfg.MODEL.DANET.MULTI_GRID = False

########################## DeepLab config ####################################
# whether to use aspp
cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP = True
# whether to use decoder
cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER = True
# whether aspp use sep conv
cfg.MODEL.DEEPLABV3_PLUS.ASPP_WITH_SEP_CONV = True
# whether decoder use sep conv
cfg.MODEL.DEEPLABV3_PLUS.DECODER_USE_SEP_CONV = True

########################## UNET config #######################################
# upsample mode
# cfg.MODEL.UNET.UPSAMPLE_MODE = 'bilinear'

########################## OCNet config ######################################
# ['base', 'pyramid', 'asp']
cfg.MODEL.OCNet.OC_ARCH = 'base'

########################## EncNet config ######################################
cfg.MODEL.ENCNET.SE_LOSS = True
cfg.MODEL.ENCNET.SE_WEIGHT = 0.2
cfg.MODEL.ENCNET.LATERAL = True


########################## CCNET config ######################################
cfg.MODEL.CCNET.RECURRENCE = 2

########################## CGNET config ######################################
cfg.MODEL.CGNET.STAGE2_BLOCK_NUM = 3
cfg.MODEL.CGNET.STAGE3_BLOCK_NUM = 21

########################## PointRend config ##################################
cfg.MODEL.POINTREND.BASEMODEL = 'DeepLabV3_Plus'

########################## hrnet config ######################################
cfg.MODEL.HRNET.PRETRAINED_LAYERS = ['*']
cfg.MODEL.HRNET.STEM_INPLANES = 64
cfg.MODEL.HRNET.FINAL_CONV_KERNEL = 1
cfg.MODEL.HRNET.WITH_HEAD = True
# stage 1
cfg.MODEL.HRNET.STAGE1.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
cfg.MODEL.HRNET.STAGE1.NUM_BLOCKS = [1]
cfg.MODEL.HRNET.STAGE1.NUM_CHANNELS = [32]
cfg.MODEL.HRNET.STAGE1.BLOCK = 'BOTTLENECK'
cfg.MODEL.HRNET.STAGE1.FUSE_METHOD = 'SUM'
# stage 2
cfg.MODEL.HRNET.STAGE2.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
cfg.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
cfg.MODEL.HRNET.STAGE2.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE2.FUSE_METHOD = 'SUM'
# stage 3
cfg.MODEL.HRNET.STAGE3.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
cfg.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
cfg.MODEL.HRNET.STAGE3.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE3.FUSE_METHOD = 'SUM'
# stage 4
cfg.MODEL.HRNET.STAGE4.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
cfg.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
cfg.MODEL.HRNET.STAGE4.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE4.FUSE_METHOD = 'SUM'

cfg.MODEL.ALIGN_CORNERS = True
cfg.MODEL.OCR.MID_CHANNELS = 512
cfg.MODEL.OCR.KEY_CHANNELS = 256
cfg.MODEL.OCR.DROPOUT = 0.05
cfg.MODEL.OCR.SCALE = 1

########################## visual config ###########################################
# visual result output dir
cfg.VISUAL.OUTPUT_DIR = r'../runs/visual/' #r'D:\Miscellaneous\picture'        #'../runs/visual/'
cfg.VISUAL.MODEL_SAVE_DIR = r'tools/runs/checkpoints/' #r'/data/Landcover/SegmenTron-master/tools/runs/checkpoints/'  #'/data/CloudShadow/SegmenTron-master/tools/runs/checkpoints/'
# log dir
cfg.VISUAL.LOG_SAVE_DIR = 'tools/runs/logs'   #r'/data/CloudShadow/SegmenTron-master/tools/runs/logs/'        #runs/logs/
cfg.VISUAL.CURRENT_NAME = ''
#cfg.TRAIN.SUMMARYWRITER_SAVE_DIR = r'E:\code\python\pytorch\my_code\Multi_models\SegmenTron-master\SegmenTron-master\tools\runs\logs'

cfg.UTILS.VAL_START = 0
cfg.UTILS.EPOCH_STOP = 1000

