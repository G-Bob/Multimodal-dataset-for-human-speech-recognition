import random

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional
import torchvision
from torchvision import transforms

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from einops import rearrange
from random import shuffle
from PIL import Image
import skvideo.io
import cv2


scalar = MinMaxScaler()

class mmwaveDataset(Dataset):

    def __init__(self, data_dir, data_list,num_classes, norm=False,
                 abs=False,img_size=(150, 150)):
        """dataset for vowel, word, sentences classification
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
        self.img_size = img_size
        self.resize = torchvision.transforms.Resize(img_size)
        self.norm = norm
        self.abs = abs
        self.mean=3.4269
        self.std =72.9885

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)

        data = np.load(file_path)
        label = int(data_file_name.split('_')[2])-1

        if self.abs:
            data = np.abs(data)
        if self.norm:
            data = stats.zscore(data)

        data = torch.tensor(data)
        data = data.type(torch.cuda.FloatTensor)
        data = data.unsqueeze(0)
        data = self.resize(data)

        label = torch.tensor(label)

        self.data = data
        self.label = label
        return data, label  # [s, d], []

class uwbDataset(Dataset):

    def __init__(self, data_dir, data_list,num_classes, norm=False,
                 abs=False,img_size=(256, 256)):
        """dataset for vowel, word, sentences classification
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
        self.img_size = img_size
        self.resize = torchvision.transforms.Resize(img_size)
        self.norm = norm
        self.abs = abs



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)

        data = np.load(file_path)
        label = int(data_file_name.split('_')[2])-1


        if self.abs:
            data = np.abs(data)
        if self.norm:
            data = stats.zscore(data)


        data = torch.tensor(data)
        data = data.type(torch.cuda.FloatTensor)
        data = data.unsqueeze(0)
        data = self.resize(data)

        label = torch.tensor(label)

        self.data = data
        self.label = label
        return data, label  # [s, d], []


class imgDataset(Dataset):

    def __init__(self, data_dir, data_list,num_classes, norm=False,
                 abs=False,img_size=(200, 200)):

        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.ToTensor(),
                                             """transforms.Normalize([0.20432955, 0.57648486, 0.86190677], [0.14605045, 0.14318292, 0.15981825])"""])
        self.norm = norm
        self.abs = abs


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)

        data = Image.open(file_path)
        data = self.transform(data)

        label = int(data_file_name.split('_')[2])-1
        label = torch.tensor(label)

        self.data = data
        self.label = label
        return data, label


class videoDataset(Dataset):
    def __init__(self, data_dir, data_list, num_classes, norm=False,
                 abs=False):

        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.norm = norm
        self.abs = abs

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)

        data = skvideo.io.vread(file_path,as_grey=True)
        label = int(data_file_name.split('_')[2]) - 1

        data = rearrange(data, 'c h w t -> t c h w')
        data = torch.tensor(data)
        data = data.type(torch.cuda.FloatTensor)

        label = torch.tensor(label)

        self.data = data
        self.label = label
        return data, label


class laserDataset(Dataset):

    def __init__(self, data_dir, data_list,num_classes, norm=False,
                 abs=False):

        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.norm = norm
        self.abs = abs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)

        data = np.load(file_path)
        label = int(data_file_name.split('_')[2])-1

        length = len(data)
        max_length = 4900
        # Pad or truncate the array as needed
        if length < max_length:
            pad_width = [(0, max_length - length)] + [(0, 0)] * (data.ndim - 1)
            data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
        elif length > max_length:
            data = data[:max_length]

        if self.abs:
            data = np.abs(data)
        if self.norm:
            data = stats.zscore(data)

        data = torch.tensor(data).unsqueeze(0)
        data = data.view(1,70,70)

        data = data.type(torch.cuda.FloatTensor)

        label = torch.tensor(label)

        self.data = data
        self.label = label
        return data, label

class uwbaudDataset(Dataset):
    def __init__(self, data_dir,uwb_list,vid_list,num_classes, norm=False,
                 abs=False, img_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir
        self.uwb_list = uwb_list
        self.vid_list = vid_list
        self.num_classes = num_classes
        self.img_size = img_size
        self.resize = transforms.Resize(img_size)
        self.norm = norm
        self.abs = abs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.20432955, 0.57648486, 0.86190677],
                                 [0.14605045, 0.14318292, 0.15981825])
        ])


    def __len__(self):
        return len(self.uwb_list)

    def __getitem__(self, index):
        uwb_file_name = self.uwb_list[index]
        uwb_file_path = os.path.join(self.data_dir,'uwb', uwb_file_name)

        aud_file_name = self.vid_list[index]
        aud_file_path = os.path.join(self.data_dir,'audio', aud_file_name)


        label = int(uwb_file_name.split('_')[2]) - 1
        # UWB data
        uwb_data = np.load(uwb_file_path)
        if self.abs:
            uwb_data = np.abs(uwb_data)
        if self.norm:
            uwb_data = stats.zscore(uwb_data)
        uwb_data = torch.tensor(uwb_data)
        uwb_data = uwb_data.type(torch.cuda.FloatTensor)
        uwb_data = uwb_data.unsqueeze(0)
        uwb_data = self.resize(uwb_data)

        aud_data = Image.open(aud_file_path)
        aud_data = self.transform(aud_data)
        aud_data = aud_data.type(torch.cuda.FloatTensor)


        label = torch.tensor(label)
        return uwb_data,aud_data,label


class uwbvidDataset(Dataset):
    def __init__(self, data_dir,uwb_list,vid_list,num_classes, norm=False,
                 abs=False, img_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir
        self.uwb_list = uwb_list
        self.vid_list = vid_list
        self.num_classes = num_classes
        self.img_size = img_size
        self.resize = transforms.Resize(img_size)
        self.norm = norm
        self.abs = abs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.20432955, 0.57648486, 0.86190677],
                                 [0.14605045, 0.14318292, 0.15981825])
        ])


    def __len__(self):
        return len(self.uwb_list)

    def __getitem__(self, index):
        uwb_file_name = self.uwb_list[index]
        uwb_file_path = os.path.join(self.data_dir,'uwb', uwb_file_name)

        vid_file_name = self.vid_list[index]
        vid_file_path = os.path.join(self.data_dir,'video', vid_file_name)

        label = int(uwb_file_name.split('_')[2]) - 1
        # UWB data
        uwb_data = np.load(uwb_file_path)
        if self.abs:
            uwb_data = np.abs(uwb_data)
        if self.norm:
            uwb_data = stats.zscore(uwb_data)
        uwb_data = torch.tensor(uwb_data)
        uwb_data = uwb_data.type(torch.cuda.FloatTensor)
        uwb_data = uwb_data.unsqueeze(0)
        uwb_data = self.resize(uwb_data)

        vid_data = skvideo.io.vread(vid_file_path, as_grey=True)
        vid_data = rearrange(vid_data, 'c h w t -> t c h w')
        vid_data = torch.tensor(vid_data)
        vid_data = vid_data.type(torch.cuda.FloatTensor)

        label = torch.tensor(label)
        return uwb_data,vid_data,label