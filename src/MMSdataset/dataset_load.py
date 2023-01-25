import os
from tkinter.tix import COLUMN
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from .common import zero_padding, load_data_BvP
from torch.nn import functional
import numpy as np
import torchvision as tv
from einops import rearrange

class multimodal_uwb(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, T_MAX, img_size=(30, 30)) -> None:
        super().__init__()
       # self.data_list = data_list
        self.T_MAX = T_MAX
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
        self.img_size = img_size
        self.resize = tv.transforms.Resize(img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data_file_name = self.data_list[index]
        data_1, label_1 = load_data_BvP(self.path_to_data, data_file_name, self.T_MAX)

        data_1 = torch.tensor(data_1)
        data_1 = rearrange(data_1, '(c h) w s -> s c h w', c=1)
        data_1 = self.resize(data_1)
        data_1 = functional.normalize(data_1, dim=0)
   
        label_1 = functional.one_hot(tensor(label_1), self.num_class).type(torch.float64)

        return data_1, label_1
    

class TimeDataset(Dataset):
    
    def __init__(self, data_dir, data_list, num_classes, col_select = "dop_spec_ToF", norm = False, down_sample=1, abs=True) -> None:
        """dataset for time data

        :param data_dir: dataset root
        :param data_list: file list of dataset 
        :param num_classes: class number
        :param col_select: select with column is using, defaults to "dop_spec_ToF"
        :param norm: if normlize data, defaults to False
        :param down_sample: down sample channels,only works for column 'timesData', defaults to 3
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.col_select = col_select
        self.norm = norm
        self.down_sample = down_sample
        self.abs = abs
    
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):
        
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)
        
        data = scio.loadmat(file_path)[self.col_select]
        label = int(data_file_name.split('-')[1]) - 1
            
        #[s, d]
        if self.abs:
         data = np.abs(data)
        data = data[::self.down_sample, :]
            
        data = torch.tensor(data)
        if self.norm:
            data = functional.normalize(data, dim = 0)
        #[]
        label = torch.tensor(label)
        
        return data, label #[s, d], []


class TimeDataset3C(Dataset):

    def __init__(self, data_dir, data_list, num_classes, col_select, norm=False, down_sample_seq=1, down_sample_width=1, down_sample_height=1, abs=True) -> None:
        """dataset for time data

        :param data_dir: dataset root
        :param data_list: file list of dataset, List[Tuple[3]]
        :param num_classes: class number
        :param col_select: select with column is using, defaults to "dop_spec_ToF"
        :param norm: if normlize data, defaults to False
        :param down_sample: down sample for image width(signal channels)
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.col_select = col_select
        self.norm = norm
        self.down_sample_seq = down_sample_seq
        self.down_sample_width = down_sample_width
        self.down_sample_height = down_sample_height
        self.abs = abs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        data = []
        label = -1
        for data_file_name in self.data_list[index]:
            file_path = os.path.join(self.data_dir, data_file_name)
            data.append(scio.loadmat(file_path)[self.col_select])
            label = int(data_file_name.split('-')[1]) - 1

        #[c, s, h, w]
        data = np.stack(data)
        
        if self.abs:
            data = np.abs(data)

        data = rearrange(data, 'c s h w -> s c h w')
        data = data[::self.down_sample_seq, :, ::self.down_sample_height, ::self.down_sample_width]

        data = torch.tensor(data)
        if self.norm:
            data = functional.normalize(data, dim=0)
        
        label = torch.tensor(label)

        return data, label  # [s, d], []

