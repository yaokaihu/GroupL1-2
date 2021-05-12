import matplotlib.pyplot as plt
import codecs
import numpy as np
import os
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

epoch = np.linspace(1,20,20)
reg_list = np.arange(0.001,0.011,0.002)
for reg in reg_list:
    path = f"./checkpoint/pre_train/mnist_lenet/mnist_lenet_0.5_1701_{reg:.3f}"
    path = os.path.join(path,"logs.txt")
    with open(path, mode='r')as f:
        line = f.readline()
        train_loss = []
        test_loss = []
        test_acc = []
        while line:
          a = line.split(",")
          train_loss.append(a[0])
          test_loss.append(a[1])
          test_acc.append(float(a[2]))
          line = f.readline()
        plt.plot(epoch,test_acc,label = reg)
plt.legend()
plt.show()

# plt.plot(epoch,base_test_acc,label = "base",color = "black")
# plt.plot(epoch,L2_test_acc,label = "L2",color = "red")
# plt.plot(epoch,L12_test_acc,label = "L12",color = "blue")
# plt.title('不同正则化惩罚 Lenet5 + MNIST')
# # 坐标轴范围
# plt.xlim((1, 20))
# plt.ylim((90, 100))
# # 坐标轴名称
# plt.xlabel('Epochs')
# plt.ylabel('test_acc')
# # 坐标轴刻度
# plt.xticks(epoch)
# plt.yticks(np.linspace(90,100,10))
# plt.legend()
# plt.savefig('./plt/test_acc.jpg')
# plt.show()
# 哦就发动i的加分加分
