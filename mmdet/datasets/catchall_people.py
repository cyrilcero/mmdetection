import mmcv
import os
import pickle
import numpy as np
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset
from torch.utils.data import Dataset
from mmdet.core import eval_map, eval_recalls
from .pipelines import Compose

@DATASETS.register_module()
class CatchAllDatasetPeople(CustomDataset):
    
    CLASSES = ('person')

    def load_annotations(self, ann_file):
        
        data_infos = []

        anno_name, ext =  os.path.splitext(ann_file)
        if ext == '.pkl':
            annos = pickle.load(open(ann_file, 'rb'))
        else:
            print('This only works if annotation file is .pkl')
        for anno in annos:
            filename = anno['filename']
            frameid = int(anno['frameid'])
            width = int(anno['width'])
            height = int(anno['height'])
            bboxes = anno['ann']['bboxes']
            labels = anno['ann']['labels']

            data_infos.append(
                dict(
                    # id=frameid,
                    filename=filename, 
                    width=width, 
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
        