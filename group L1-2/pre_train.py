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
    pre_path = f'./checkpoint/pre_train/{args.dataset}_{args.network}/{args.dataset}_{args.network}_{args.penalty}_{args.seed}_{args.reg_param:.3F}'
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



    #
    # '''prune'''
    # # 新建prune文件夹
    # pru_path = f'./checkpoint/pruned/{args.dataset}_{args.network}/{args.dataset}_{args.network}_{args.penalty:.1f}_{args.seed}_{args.reg_param}_pruned'
    # if not os.path.exists(pru_path):
    #     os.makedirs(pru_path)
    #
    # # 阈值剪枝后，存储conv权重
    # with open(os.path.join(pru_path, "weight.txt"), "w")as f:
    #     for name, param in network.named_parameters():
    #         if 'conv' in name and 'weight' in name:
    #             # 对filter
    #             for i in range(param.shape[0]):
    #                 param.data[i, :, :, :] = zero_out(param.data[i, :, :, :], args.thre)
    #                 # 如果该filter的L1范数太小，则零化该过滤器
    #                 if torch.sum(abs(param.data[i, :, :, :])).item() < 1:
    #                     param.data[i, :, :, :] = 0
    #                 print(torch.sum(abs(param.data[i, :, :, :])).item())
    #         if 'fc' in name and 'weight' in name:
    #             # 对fc
    #             param.data = zero_out(param.data, args.thre)
    #         f.write(f"{name}{param}")
    # print("pruned")
    #
    # # 存储模型
    # torch.save(network.state_dict(), os.path.join(pru_path, 'model.pth'))


if __name__ == '__main__':
    main()
