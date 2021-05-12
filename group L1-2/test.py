
import torch
from network import *
import numpy as np
import matplotlib.pyplot as plt

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



network = LeNet5().cuda()

state_dict = torch.load(f'./checkpoint/pruned/mnist_lenet/mnist_lenet_2.0_1701_0.005_pruned/model.pth')
network.load_state_dict(state_dict)

for name, param in network.named_parameters():
    if 'conv' in name and 'weight' in name:
        for i in range(param.shape[0]):
            print(torch.sum(abs(param.data[i, :, :, :])).item())