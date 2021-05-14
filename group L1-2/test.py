
import torch
from network import *
import numpy as np
import matplotlib.pyplot as plt

def get_sparsity(ith_filter):

    p = ith_filter.cpu().detach().numpy()
    nz_cout = np.count_nonzero(p)
    total_count = p.size

    return round(100 * (1 - nz_cout / total_count), 1)

'''fileter权重均值'''
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# network = LeNet5()
# conv1 = []
# conv2 = []
# for i in range(1,10):
#     state_dict = torch.load(f'./checkpoint/pre_train/mnist_lenet/mnist_lenet_0.5_1701_0.0{i}0/model.pth')
#     network.load_state_dict(state_dict)
#
#     for name, param in network.named_parameters():
#         if "conv1.weight"in name:
#             conv1.append(np.mean(param.detach().numpy()))
#
#         if "conv2.weight"in name:
#             conv2.append(np.mean(param.detach().numpy()))
# print(conv1)
# print(conv2)
# x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
# plt.plot(x,conv1)
# for a, b in zip(x, conv1):
#     plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
# plt.title("正则化参数取值效果")
# plt.show()


'''保留的filter中零占比'''
if __name__ == '__main__':
    network = LeNet5().cuda()

    reg_list = np.arange(0.001, 0.021, 0.001)
    for reg_param in reg_list:
        print(f"reg_param:{reg_param:.3f}")
        state_dict = torch.load(f'./checkpoint/pruned/mnist_lenet/mnist_lenet_0.5_1701_{reg_param:.3f}_pruned/model.pth')
        network.load_state_dict(state_dict)

        for name, param in network.named_parameters():
            if 'conv' in name and 'weight' in name:
                print(name)
                for i in range(param.shape[0]):
                    ith_filter_L1_score = torch.sum(abs(param.data[i, :, :, :])).item()
                    if ith_filter_L1_score > 0:
                        print(i,  get_sparsity(param.data[i, :, :, :]))

