import torch.nn as nn
import torch.nn.functional as F


# 定义 Convolution Network 模型
# He初始化权重,在hyperparameter.py中改为xaiver初始化
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,20,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(20, 50, 5)

        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# model = LeNet5()
# #打印某一层的参数名
# for name in model.state_dict():
#     print(name)
# # 获取第一层卷积权重
# conv1 = model.state_dict()['conv1.weight']
# print(conv1.shape)
# # 获取所有权重
# with open("../weight.txt","w")as f:
#     for name, param in model.named_parameters():
#         f.write(f"{name}{param}")

