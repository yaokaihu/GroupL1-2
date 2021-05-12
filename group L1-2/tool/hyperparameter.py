'''获取训练参数'''
from functools import partial
from network import *
import torch.nn as nn

import torch.optim as optim


def get_hyperparameters(network_type):

	if network_type == 'lenet':
		network = LeNet5().cuda()
		# 讲He初始化 改为 Xaiver初始化
		for m in network.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear)):
				nn.init.xavier_uniform_(m.weight)
		optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005)
		train_iteration = 10000
		test_iter = 500
		# finetune_iteration = 5000
		train_batch_size = 64
		test_batch_size =100
		stepvalue = [5000, 7000, 8000, 9000, 9500]

	elif network_type == 'mlp':
		network = mlp().cuda()
		optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005)
		train_iteration = 20000
		test_iter = 500
		# finetune_iteration = 50000
		train_batch_size = 64
		test_batch_size =100
		stepvalue = [10000, 14000, 18000]
	else:
		raise ValueError('Unknown network')

	return network, optimizer, train_iteration, train_batch_size, test_batch_size, test_iter, stepvalue