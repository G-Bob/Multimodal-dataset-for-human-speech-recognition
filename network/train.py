import os
import sys
import argparse
import csv
import pandas as pd
import time
from datetime import datetime
from random import shuffle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader

import loaddataset,conf

def train(epoch,x,train_loss_list,train_acc_list,val_loss_list,val_acc_list):

    start = time.time()
    x.append(epoch)
    train_loss = 0.0
    val_loss = 0.0
    correct_train = 0.0
    correct_val = 0.0
    net.train()
    for batch_index, (data, labels) in enumerate(train_loader):
        labels = labels.cuda()
        data = data.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct_train += preds.eq(labels).sum()
        loss.backward()
        optimizer.step()
    train_loss_list.append(train_loss / len(train_loader.dataset))
    train_acc_list.append(correct_train.cpu() / len(train_loader.dataset))
    scheduler.step()

    torch.cuda.empty_cache()
    net.eval()
    for (data, labels) in test_loader:
        data = data.cuda()
        labels = labels.cuda()

        outputs = net(data)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct_val += preds.eq(labels).sum()
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.9f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        epoch=epoch
    ))
    val_loss_list.append(val_loss / len(test_loader.dataset))
    val_acc_list.append(correct_val.cpu() / len(test_loader.dataset))
    # update training loss for each iteration
    finish = time.time()
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    accuracy = correct_val / len(test_loader.dataset)
    print('Average train loss: {:.4f}'.format(train_loss))
    print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        val_loss,
        accuracy,
    ))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    plt.close()
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(x, train_loss_list, 'r')
    ax1.plot(x, val_loss_list, 'b')
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    ax2 = fig.add_subplot(222)
    ax2.plot(x, train_acc_list, 'g')
    ax2.plot(x, val_acc_list, 'y')
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.draw()
    plt.pause(1)

    torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training the THAT model with timedata dataset")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--nclass', type=int, default=15)
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--data_dir', default=r'.\distance_dataset\24\2')
    parser.add_argument('--save_dir', default=r'.\parms\distance')
    parser.add_argument('--net', type=str, default='resnet18')
    args = parser.parse_args()
    device = torch.device(args.device)

    net = conf.get_network(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(1000, 1000, 10), gamma=0.5)

    datalist = os.listdir(args.data_dir)
    train_list_new = []
    test_list_new = []

    path_train = os.path.join (args.data_dir,'train')
    path_test = os.path.join(args.data_dir,'test')

    train_list = os.listdir(path_train)
    test_list = os.listdir(path_test)

    if args.env:
        for train_file in train_list:
            for x in args.env:
                if int(train_file.split('_')[0]) == x:
                    train_list_new.append(train_file)
        for test_file in test_list:
            for x in args.env:
                if int(test_file.split('_')[0]) == x:
                    test_list_new.append(test_file)

        train_list = train_list_new
        test_list = test_list_new

    shuffle(test_list),shuffle(train_list)

    train_data = loaddataset.uwbDataset(path_train, train_list, num_classes=args.nclass, norm=True, abs=True)
    test_data = loaddataset.uwbDataset(path_test, test_list, num_classes=args.nclass, norm=True, abs=True)

    train_loader = DataLoader(train_data, batch_size=args.batch)
    test_loader = DataLoader(test_data,batch_size=args.batch)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    j = 0
    x = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []


    for i in range(0, args.epoch):

        train(i, x, train_loss_list, train_acc_list, val_loss_list, val_acc_list)


    torch.cuda.empty_cache()






