import numpy as np
import argparse
import torch
import os,sys

from random import shuffle


path_data = "/home/wang/wjy/Multimodal_dataset_validation/dataset/multimodal_uwb/"
filename = "1_1_1_1_sample1.npy"

data = np.load(os.path.join(path_data,filename),"r")

    
def random_split_data_list(data_list: list, test_ratio):
    shuffle(data_list)
    test_size = round(len(data_list)*test_ratio)
    train_size = len(data_list) - test_size
    train_list = data_list[:train_size]
    test_list = data_list[train_size:]
    return train_list, test_list


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="training the THAT model with timedata dataset")
    """
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--nclass', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=124)
    parser.add_argument('--input_dim', type=int, default=61, help='the dim of model\'s inpu')
    parser.add_argument('--n_seq', type=int, default=300, help='the number of channel of the model\'s input')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--hlayers', type=int, default=4)
    parser.add_argument('--vlayers', type=int, default=4)
    parser.add_argument('--hheads', type=int, default=4)
    parser.add_argument('--vheads', type=int, default=4)
    

    parser.add_argument('--log', type=int, default=0, help='a bool value, if true, will print output to a log file, default=0')
    parser.add_argument('--col_select', default='dop_spec_ToF')
    parser.add_argument('--down_sample', type=int, default=1, help='down sample for channels')
    """
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data_dir', default='/home/wang/wjy/Multimodal_dataset_validation/dataset/multimodal_uwb')
    # parser.add_argument('--save_dir', default='./../models/THAT_Timedata')
    args = parser.parse_args()
    
    device = torch.device(args.device)
        
    sys.path.append(r'/home/wang/wjy/Multimodal_dataset_validation/src/MMSdataset')

    """prepare data
    """
    list = os.listdir(args.data_dir)
    
    args.env = 1 # Choose the index of volunteer
    #select environment
    if args.env:
        list_new = []
        for file in list:
            if int(file.split('_')[0]) == args.env:
                list_new.append(file)
        
        list = list_new
        
    train_list, test_list = random_split_data_list(list, 0.2)
    
