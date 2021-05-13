import numpy as np
from numpy import *
import torch

def zero_out(X,thre):
    orig_shape = X.shape
    X = np.array(X.flatten().cpu().detach())
    zero_out_idx = nonzero(abs(X)<thre)   # 返回非零元素的索引
    X[zero_out_idx] = 0
    return torch.tensor(np.reshape(X,orig_shape))

def get_sparsity(model):
	nonzero = 0
	total = 0
	for name, param in model.state_dict().items():
		if 'weight' in name: # 因为只对weight结构稀疏化
			p = param.cpu().detach().numpy()
			nz_count = np.count_nonzero(p)
			total_count = p.size
			nonzero += nz_count
			total += total_count
    #
	# 		print(f'{name:20} | nonzeros = {nz_count:7}/{total_count} ({100 * nz_count / total_count:6.2f}%) | total_pruned = {total_count - nz_count:7} | shape= {list(p.data.shape)}')
	# print(f'surv: {nonzero}, pruned: {total - nonzero}, total: {total}, Comp. rate: {total / nonzero:10.2f}x ({100 * (total - nonzero) / total:6.2f}% pruned)')

	return 100 * (1 - nonzero / total)
