'''获取数据集'''

import torchvision.datasets as datasets
import torchvision.transforms as T


def get_dataset(dataset, normalize=True):
	if dataset == 'mnist':
		# 将数据范围归一化为(0,1)
		transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]) if normalize else T.ToTensor()
		train_dataset = datasets.MNIST('./dataset', train=True, download=True, transform=T.ToTensor())
		test_dataset = datasets.MNIST('./dataset', train=False, download=True, transform=T.ToTensor())
	return train_dataset, test_dataset
