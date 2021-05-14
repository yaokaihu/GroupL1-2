'''train、test'''

import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tool.regularization import Regularization

'''The dataset is divided into several batch_size according to the num_iterations '''


class BatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations


def train(train_dataset, test_dataset, network, optimizer, num_iterations=10000,
          train_batch_size=64, test_batch_size=100, test_iter=500, penalty=0.5, reg_param=0.005, stepvalue=[]):
    print(reg_param)
    network.train()
    batch_sampler = BatchSampler(train_dataset, num_iterations, train_batch_size)  # train by iteration, not epoch
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
    optimizer = optimizer(network.parameters())
    sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, stepvalue, gamma=0.9,
                                                 last_epoch=-1)
    sum = []
    if sched is not None:
        for i, (x, y) in enumerate(train_loader):
            print(i)
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            out = network(x)
            loss = F.cross_entropy(out, y)
            reg_loss = Regularization(network, weight_decay=reg_param).to("cuda")
            loss += reg_loss(network, penalty)
            loss.backward()
            optimizer.step()
            sched.step()

            # 每500次进行一次test
            if (i + 1) % test_iter == 0:
                # print(network.state_dict()['conv1.weight'])
                test_acc, test_loss = test(network, test_dataset, test_batch_size)
                print(f'Steps: {i + 1}/{num_iterations}\tTest loss: {test_loss:.3f}\tTest acc: {test_acc:.2f}')
                network.train()  # train mode
                # 记录 sum中每行包括loss、test_loss、test_acc
                sum.append([])
                count = int(((i + 1) / test_iter) - 1)
                sum[count].append(loss.item())
                sum[count].append(test_loss)
                sum[count].append(test_acc)

    return sum


def test(network, dataset, batch_size=100):
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    loss = 0
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            out = network(x)
            _, pred = out.max(1)

        correct += pred.eq(y).sum().item()
        loss += F.cross_entropy(out, y) * len(x)

    acc = correct / len(dataset) * 100.0
    loss = loss.item() / len(dataset)

    return acc, loss
