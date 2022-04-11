"""
This module provides data loaders and transformers for popular vision datasets.
"""
#rom .naic_rsimage import NAICSegmentation
from .cloud_shadow import CloudShadowSegmentation
from .landcover_9num import Landcover9num_Segmentation
from .gdal_landcover_9num import GDALLandcover9num_Segmentation
from .gdal_landcover30 import GDALLandcover_Segmentation
from .gdal_landcover30_binary import GDALBinary_Segmentation
from .gdal_landcover_common import GDALLandcoverCommon_Segmentation
from .dataset_vaihingen_potsdam import VaihingenPotsdamDataset
datasets = {
    #'naic': NAICSegmentation,
    'cloudshadow':CloudShadowSegmentation,
    'landcover9num':Landcover9num_Segmentation,
    'gdal_landcover':GDALLandcover_Segmentation,
    'gdal_binary':GDALBinary_Segmentation,
    'gdal_landcover_common': GDALLandcoverCommon_Segmentation,
    'vaihingenpotsdam_dataset': VaihingenPotsdamDataset
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
