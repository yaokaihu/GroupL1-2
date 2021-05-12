import torch
from tool.hyperparameter import get_hyperparameters
from tool.train import test
from tool.dataset import get_dataset
from tool.sparsity import *
from tool.sparsity import get_sparsity
import os


def main():
    dataname = "mnist"
    netname = "lenet"
    penalty = 0.5
    seed = 1701
    lasso_network, optimizer, train_iteration, train_batch_size, test_batch_size, test_iter, stepvalue = get_hyperparameters(
        netname)
    network, optimizer, train_iteration, train_batch_size, test_batch_size, test_iter, stepvalue = get_hyperparameters(
        netname)

    # 新建data_net文件夹
    path = f'./checkpoint/contrast/{dataname}_{netname}'
    if not os.path.exists(path):
        os.makedirs(path)

    train_dataset, test_dataset = get_dataset(dataname)

    # 加载pruned group lasso模型
    laaso_state_dict = torch.load(
        f'./checkpoint/pruned/{dataname}_{netname}/{dataname}_{netname}_2.0_{seed}_0.005_pruned/model.pth')
    lasso_network.load_state_dict(laaso_state_dict)

    # group lasso测试
    lasso_test_acc, lasso_test_loss = test(lasso_network, test_dataset)
    lasso_sparsity = get_sparsity(lasso_network)

    # group L1/2测试
    reg_list = np.round(np.arange(0.001, 0.021, 0.001),3)
    for reg_param in reg_list:
        state_dict = torch.load(
            f'./checkpoint/pruned/{dataname}_{netname}/{dataname}_{netname}_0.5_{seed}_{reg_param:.3f}_pruned/model.pth')  # 加载pruned group L1/2模型
        network.load_state_dict(state_dict)
        test_acc, test_loss = test(network, test_dataset)
        sparsity = get_sparsity(network)

        with open(os.path.join(path,"0.5.txt"),"a")as f:
            f.write(f"reg:{reg_param:.3f}\t{lasso_test_acc:.2f},{test_acc:.2f}\t\t{lasso_sparsity:.2f},{sparsity:.2f}\n")
        print("test")


if __name__ == '__main__':
    main()
