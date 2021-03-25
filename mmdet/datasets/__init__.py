from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .custom import CustomDataset
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               RepeatDataset)
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .catchall import CatchAllDataset
<<<<<<< HEAD
from .catchall_lpr import CatchAllDatasetLPR
from .catchall_people import CatchAllDatasetPeople
=======
>>>>>>> 073b83c13482814749827922a80ac4abd60e265e

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 
    'VOCDataset', 'CityscapesDataset', 'LVISDataset', 'GroupSampler',
    'DistributedGroupSampler', 'DistributedSampler', 'build_dataloader',
    'ConcatDataset', 'RepeatDataset', 'ClassBalancedDataset',
    'WIDERFaceDataset', 'DATASETS', 'PIPELINES', 'build_dataset',
<<<<<<< HEAD
    'CatchAllDatasetLPR', 'CatchAllDataset', 'CatchAllDatasetPeople',
=======
    'CatchAllDataset'
>>>>>>> 073b83c13482814749827922a80ac4abd60e265e
]
