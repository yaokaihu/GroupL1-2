'''正则化惩罚'''
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.conv_weight_list, self.fc_weight_list = [],[]

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        conv_weight_list = []
        fc_weight_list = []
        for name, param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                weight = (name, param)
                conv_weight_list.append(weight)
            if 'fc' in name and 'weight' in name:
                weight = (name, param)
                fc_weight_list.append(weight)
        return conv_weight_list, fc_weight_list

    def regularization_loss(self, conv_weight_list,penalty):
        '''group lasso  or  group L1/2'''
        reg_loss = 0
        filter_reg_loss = 0
        channel_reg_loss = 0
        p = 1 if penalty == 0.5 else 2
        for name, w in conv_weight_list:
            # filter级,如：lenet5的conv1为[20,1,5,5],就是有20个filter
            for i in range(w.shape[0]):
                ith_filter_weight = w[i, :, :, :]  # 第i个filter权重
                ith_filter_reg_loss = torch.sqrt(torch.norm(ith_filter_weight, p=p))
                filter_reg_loss += ith_filter_reg_loss
            # channel级
            for i in range(w.shape[1]):
                ith_channel_weight = w[:, i, :, :]  # 第i个channel权重
                ith_channel_reg_loss = torch.sqrt(torch.norm(ith_channel_weight, p=p))
                channel_reg_loss += ith_channel_reg_loss

        reg_loss = self.weight_decay * filter_reg_loss + self.weight_decay * channel_reg_loss
        return reg_loss
    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model,penalty):
        self.conv_weight_list, self.fc_weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.conv_weight_list, penalty)
        return reg_loss