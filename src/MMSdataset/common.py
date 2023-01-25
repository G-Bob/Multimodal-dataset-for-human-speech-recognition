from importlib.resources import path
import os
from socketserver import DatagramRequestHandler
import numpy as np
#import scipy.io as scio
import torch.nn as nn
from random import shuffle


def random_split_data_list(data_list: list, test_ratio):
    shuffle(data_list)
    test_size = round(len(data_list)*test_ratio)
    train_size = len(data_list) - test_size
    train_list = data_list[:train_size]
    test_list = data_list[train_size:]
    return train_list, test_list

