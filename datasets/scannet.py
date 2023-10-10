"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_20, VALID_CLASS_IDS_200

max_val =  np.array([[93.11168915, 92.34787068, 99.93811643]])
min_val = np.array([[-96.98585108, -94.20407562, -99.97897124]])
max_min_diff =  np.array([[190.09754023, 186.55194631, 199.91708767]])
mean =  np.array([[0.51342754, 0.50512136, 0.48496683]])
std =  np.array([[0.07366831, 0.074773, 0.09917635]])


@DATASETS.register_module()
class ScanNetDataset(Dataset):

    def __init__(self,
                 split='train',
                 data_root='data/pipe',
                 class_names=None,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 cache_data=False,
                 loop=1):
        super(ScanNetDataset, self).__init__()
        print(data_root, class_names)
        self.data_root = data_root
        self.data_dir = data_root
        #self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1  # force make loop = 1 while in test mode
        self.cache_data = cache_data
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        self.num_points = 4096

        if self.split == "train":
            self.data_list = glob.glob(os.path.join(self.data_root,"train", "*.npy"))
        
        elif self.split == "val":
            self.data_list = glob.glob(os.path.join(self.data_root,"test", "*.npy"))

        elif self.split == "test":
            self.data_list = glob.glob(os.path.join(self.data_root,"test", "*.npy"))

        else:
            raise NotImplementedError

        logger = get_root_logger()

      
    def get_data(self, idx):

        data_path = self.data_list[idx % len(self.data_list)]

        np_file_points = np.load(data_path)

        data = np_file_points[:,0] # shape (4096,3)
        label = np_file_points[:,1] # shape (4096,3)
        
        '''
        data = np.expand_dims(data, axis = 0)#  shape (1,4096,3)
        label = np.expand_dims(label, axis = 0) # shape (1,4096,3)

        data = (data - min_val)/max_min_diff # shape (1,4096,3)
        data = (data - mean)/std # shape (1,4096,3)

        label = (label - min_val)/max_min_diff # shape (1,4096,3)
        label = (label - mean)/std # shape (1,4096,3)

        data = np.squeeze(data, axis=0) # shape (4096,3)
        label = np.squeeze(label, axis=0) # shape (4096,3)
        '''
        data_dict = dict(coord=data, segment=label)
        # print(len(data_dict))
        return data_dict
        

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict


    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict
        
    def get_data_name(self, idx):
        return "Phase2"
    def __getitem__(self, idx):

        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    class2id = np.array(VALID_CLASS_IDS_200)

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt200" in data.keys():
            segment = data["semantic_gt200"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            segment[sampled_index] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict