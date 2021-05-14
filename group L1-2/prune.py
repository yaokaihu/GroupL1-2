import torch
from tool.hyperparameter import get_hyperparameters
from tool.sparsity import *
import os
import math


def decimalPrecision(x):
    '''
    小数的精度
    :param x: list
    :return:
    '''
    a = []
    for i in x:
        if i == 0:
            a.append(0)
            continue
        a.append(abs(int(math.log10(abs(i)))))
    return a


def filter_L1_score(x):
    """filter的L1范数"""
    return torch.sum(abs(x)).item()

def removeFilter(ele,param):
    '''
    剪枝L1范数小的filter
    :param ele: 小数精度列表
    :param param: filter:[20,1,5,5]
    :return:
    '''
    dec = decimalPrecision(ele)
    major = majorityElement(dec)  # 以L1得分的小数精度的众数，定位到要剪枝的filter
    for i in range(param.shape[0]):
        if dec[i] >= major:
            param.data[i, :, :, :] = torch.zeros_like(param.data[i, :, :, :])  # 剪枝filter

def majorityElement(nums):
    """众数"""
    if len(nums) == 1:
        return nums[0]
    dict = {}
    for i in nums:
        if i in dict:  # 在dict是否已存在键i
            dict[i] += 1
            if dict.get(i) >= (len(nums) + 1) / 2:
                return i
        else:
            dict[i] = 1


if __name__ == '__main__':
    dataname = "mnist"
    netname = "lenet"
    network, optimizer, train_iteration, train_batch_size, test_batch_size, test_iter, stepvalue = get_hyperparameters(
        netname)

    # 阈值
    thre = 0.0001
    penalty = 2
    seed = 1701
    # reg_list = np.arange(0.001, 0.021, 0.001)

    reg_list = [0.005]
    for reg_param in reg_list:
        # 加载pre_train模型
        state_dict = torch.load(
            f'./checkpoint/pre_train/{dataname}_{netname}/{dataname}_{netname}_{penalty:.1f}_{seed}_{reg_param:.3f}/model.pth')
        network.load_state_dict(state_dict)
        # 新建剪枝文件夹
        path = f'./checkpoint/pruned/{dataname}_{netname}/{dataname}_{netname}_{penalty:.1f}_{seed}_{reg_param:.3f}_pruned'
        if not os.path.exists(path):
            os.makedirs(path)

        # 阈值剪枝后，存储conv权重
        with open(os.path.join(path, "weight.txt"), "w")as f:
            with open(os.path.join(path, "conv_L1_score.txt"), "w")as f1:
                for name, param in network.named_parameters():
                    if 'conv' in name and 'weight' in name:
                        ele = []
                        for i in range(param.shape[0]):
                            param.data[i, :, :, :] = zero_out(param.data[i, :, :, :], thre)  # 以阈值剪枝weight
                            ith_filter_L1_score = filter_L1_score(param.data[i, :, :, :])   # 第i个filter的L1得分
                            ele.append(ith_filter_L1_score)

                        removeFilter(ele,param)

                        # 记录剪枝后的filterL1范数
                        for i in range(param.shape[0]):
                            ith_filter_L1_score = filter_L1_score(param.data[i, :, :, :])   # 第i个filter的L1得分
                            f1.write(f'{ith_filter_L1_score}\n')

                    if 'fc' in name and 'weight' in name:
                        param.data = zero_out(param.data, thre)
                    f.write(f"{name}{param}")

        print("pruned")

        # 存储模型
        torch.save(network.state_dict(), os.path.join(path, 'model.pth'))
