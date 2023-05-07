import os
import sys
import re
import datetime

import numpy

import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

radar_mean = (3.4269)
radar_std = (72.9885)

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'resnet1d':
        from models.resnet1d import resnet1d
        net = resnet1d()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net =='r3d_18':
        from models.resnetvideo import r3d_18
        net = r3d_18()
    elif args.net =='multiresnet18':
        from models.multicnn import multiresnet18
        net = multiresnet18()
    elif args.net == 'dualresnet18':
        from models.dual_resnet import dualresnet18
        net = dualresnet18()
    elif args.net == 'dualresnet3d':
        from models.dual_resnet3d import dualresnet3d
        net = dualresnet3d()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net




