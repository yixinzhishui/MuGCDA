"""
This module provides data loaders and transformers for popular vision datasets.
"""
#rom .naic_rsimage import NAICSegmentation
from .dataset_vaihingen_potsdam import VaihingenPotsdamDataset
datasets = {
    #'naic': NAICSegmentation,
    'vaihingenpotsdam_dataset': VaihingenPotsdamDataset
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
