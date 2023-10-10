
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


max_val =  np.array([[93.11168915, 92.34787068, 99.93811643]])
min_val = np.array([[-96.98585108, -94.20407562, -99.97897124]])
max_min_diff =  np.array([[190.09754023, 186.55194631, 199.91708767]])
mean =  np.array([[0.51342754, 0.50512136, 0.48496683]])
std =  np.array([[0.07366831, 0.074773, 0.09917635]])


def create_data_frame(data_path, group_type):

    if group_type == 'train':
        data_dir = os.path.join(data_path, group_type)
        # data_dir = os.path.join(data_path,'single_radius_pipe_dataset_large_train/numpy_all_points')
        
    else:
        data_dir = os.path.join(data_path, 'test')
        # data_dir = os.path.join(data_path,'single_radius_pipe_dataset_large_test/numpy_all_points')

    data_batchlist, label_batchlist = [], []
    np_files = sorted(os.listdir(data_dir))
    
    for f in np_files:

        np_file_points = np.load(os.path.join(data_dir, f))

        data = np_file_points[:,0] # shape (4096,3)
        label = np_file_points[:,1] # shape (4096,3)
        
        data = np.expand_dims(data, axis = 0)#  shape (1,4096,3)
        label = np.expand_dims(label, axis = 0) # shape (1,4096,3)

        # make each value 0 to 1 and then 0 mean 1 std
        data = (data - min_val)/max_min_diff # shape (1,4096,3)
        data = (data - mean)/std # shape (1,4096,3)

        label = (label - min_val)/max_min_diff # shape (1,4096,3)
        label = (label - mean)/std # shape (1,4096,3)
        
        data_batchlist.append(data)
        label_batchlist.append(label)

    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)

    return data_batches, seg_batches

@DATASETS.register_module()
class PipeDataset(Dataset):

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
            self.data, self.seg = create_data_frame(self.data_root, self.split)
        
        elif self.split == "val":
            self.data, self.seg = create_data_frame(self.data_root, self.split)

        elif self.split == "test":
            self.data, self.seg = create_data_frame(self.data_root, self.split)

        else:
            raise NotImplementedError

        logger = get_root_logger()

        

    def get_data_list(self):
        raise NotImplementedError
      
    def get_data(self, idx):
        
        pointcloud = self.data[idx][:self.num_points] # (4096x3)
        seg = self.seg[idx][:self.num_points] # (4096x3)
        
        data_dict = dict(coord=pointcloud, segment=seg)
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
        return self.data.shape[0]