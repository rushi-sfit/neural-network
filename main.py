#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import matplotlib
import torchvision
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.nn.utils import prune

from prune import pruner

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os


from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNN
from models.Fed import FedAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # parse args
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
   #"cifar_no_dp/yuan_clip20"
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    start_time = time.time()
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=False, transform=trans_mnist)

        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=False, transform=trans_mnist)


        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model

    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN()

    else:
        exit('Error: unrecognized model')


    dp_epsilon = args.dp_epsilon / (args.frac * args.epochs)
    print("per train:{}".format(dp_epsilon))
    dp_delta = args.dp_delta
    dp_mechanism = args.dp_mechanism
    dp_clip = args.dp_clip

    print(net_glob)
    #是否使用剪枝
    # pruner(net_glob)

    net_glob.train()


    # to verify that all masks exist
    # copy weights

    r_x_train = []
    r_x_test = []
    all_clients = list(range(args.num_users))
    best_acc = -1
    # training
    acc_test = []
    loss_all=[]
    time1=[]
    learning_rate = [args.lr for i in range(args.num_users)]
    for iter in range(args.epochs):

        w_locals, loss_locals = [], []
        w_glob = net_glob.state_dict()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)
        begin_index = iter % (1 / args.frac)
        idxs_clients = all_clients[int(begin_index * args.num_users * args.frac):
                                   int((begin_index + 1) * args.num_users * args.frac)]
        for idx in idxs_users:
            args.lr = learning_rate[idx]


            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                dp_epsilon=dp_epsilon, dp_delta=dp_delta,dp_clip=dp_clip,
                                dp_mechanism=dp_mechanism)
            w, loss, curLR = local.train(net=copy.copy(net_glob).to(args.device))
            learning_rate[idx] = curLR

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights

        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # pruner(net_glob)

        # print accuracy
        net_glob.eval()

        loss_t,acc_t = test_img(net_glob, dataset_test, args)
        if acc_t>=94.00:
            print("Best epoch:{}".format(iter))
        if best_acc < acc_t:
            torch.save(net_glob, 'resnet18-round%d.pth' % (args.local_ep))
            best_acc = acc_t
        tim1 = time.time() - start_time
        print("Round {:3d},loss {:3f},Testing accuracy: {:.6f},time: {:.2f}".format(iter, loss_t, acc_t, tim1))

        loss_all.append(loss_t)
        acc_test.append(acc_t)
        time1.append(tim1)
    print("Best Acc=%.4f" % (best_acc))
    with open("fashion_2_dp_loss_xin10niid.txt", 'w') as train_los:
        train_los.write(str(loss_all))
    with open("fashion_2_dp_acc_xin10niid.txt", 'w') as train_ac:
        train_ac.write(str(acc_test))
    with open("fashion_2_dp_acc_xin10niid_time.txt", 'w') as tim:
        tim.write(str(time1))





