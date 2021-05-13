'''主函数'''

import argparse
import os
import random
import numpy as np

import torch
import tool

from tool.dataset import get_dataset
from tool.train import train, test
from tool.hyperparameter import get_hyperparameters
from tool.sparsity import *


def main():
    '''获取参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1701, help='random seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset (mnist|fmnist|cifar10)')
    parser.add_argument('--network', type=str, default='lenet', help='network (mlp|lenet|conv6|vgg19|resnet18)')
    parser.add_argument('--penalty', type=float, default='2', help='regularization type(0.5|2)')
    parser.add_argument('--reg_param', type=float, default='0.005', help='regularization parameter')
    parser.add_argument('--thre', type=float, default='0.0001', help='Threshold value of pruning weight')
    args = parser.parse_args()

    '''固定随机值'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次相同输入，有固定的相同输出
    torch.backends.cudnn.benchmark = False

    '''pre_train'''

    # 训练
    train_dataset, test_dataset = get_dataset(args.dataset)

    network, optimizer, train_iteration, train_batch_size, test_batch_size, test_iter, stepvalue = get_hyperparameters(
        args.network)

    # 新建pre_train文件夹
    pre_path = f'./checkpoint/pre_train/{args.dataset}_{args.network}/{args.dataset}_{args.network}_{args.penalty:.1f}_{args.seed}_{args.reg_param:.3F}'
    # pre_path = f'./checkpoint/pre_train/{args.dataset}_{args.network}/baseline'

    if not os.path.exists(pre_path):
        os.makedirs(pre_path)
    result = train(train_dataset, test_dataset, network, optimizer, train_iteration, train_batch_size,
                   test_batch_size, test_iter, args.penalty, args.reg_param, stepvalue)

    # 存储
    torch.save(network.state_dict(), os.path.join(pre_path, 'model.pth'))
    with open(os.path.join(pre_path, 'logs.txt'), 'w') as f:
        for i in result:
            # train_loss      test_loss     test_acc
            f.write(f"{i[0]:.3f},{i[1]:.3f},{i[2]:.3f}\n")
    with open(os.path.join(pre_path, 'weight.txt'), "w")as f:
        for name, param in network.named_parameters():
            f.write(f"{name}{param}")

if __name__ == '__main__':
    main()
