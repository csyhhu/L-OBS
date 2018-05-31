""" 
This code prune weights in ResNet18 usin L-OBS
""" 
import os
import numpy as np
from datetime import datetime
import cPickle

from utils import unfold_kernel, fold_weights, get_error

import torch

# data path specify
hessian_inverse_root = './ResNet18/hessian_inv'
hessian_inverse_Woodbury_root = './ResNet18/hessian_inv'
save_root = './ResNet18/pruned_weight'

pretrain = torch.load('./ResNet18/resnet18-5c106cde.pth')
# Layer name of ResNet
layer_name_list = list()
layer_name_list.append('fc')
for layer_name in pretrain.keys():
    if 'conv1.weight' in layer_name or 'conv2.weight' in layer_name or 'conv3.weight' in layer_name \
        or 'downsample.0.weight' in layer_name:
        layer_name_list.append(layer_name[: -7])

for layer_idx, layer_name in enumerate(layer_name_list):

	# if os.path.exists('%s/CR_5/%s.weight.npy' %(save_root, layer_name)):
	# 	continue

	print ('[%s] %s' %(datetime.now(), layer_name))
	# Specify layer type, C for convolution, F for fully-connected
	if layer_name == 'fc':
		layer_type = 'F'
	else:
		layer_type = 'R'

	if layer_type == 'C':
		kernel = pretrain['%s.weight' %layer_name].data.numpy()
		kernel_shape = kernel.shape # [64, 3, 11, 11]
		weight = unfold_kernel(kernel) # [364, 64]
		bias = pretrain['%s.bias' %layer_name].data.numpy() # [64, ]
		wb = np.concatenate([weight, bias.reshape(1, -1)], axis = 0) # [365, 64]
	elif layer_type == 'F':
		weight = pretrain['%s.weight' %layer_name].data.numpy()
		bias = pretrain['%s.bias' %layer_name].data.numpy()
		wb = np.hstack([weight, bias.reshape(-1, 1)]).transpose()
	elif layer_type == 'R':
		kernel = pretrain['%s.weight' %layer_name].data.numpy()
		kernel_shape = kernel.shape # [64, 3, 11, 11]
		wb = unfold_kernel(kernel) # [364, 64]

	# hessian = np.load('%s/%s.npy' %(hessian_root, layer_name))
	if layer_name == 'fc':
		hessian_inv = np.load('%s/%s.npy' %(hessian_inverse_Woodbury_root, layer_name))
	else:
		hessian_inv = np.load('%s/%s.npy' %(hessian_inverse_root, layer_name))

	l1, l2 = wb.shape
	# mask = np.ones(wb.shape)

	# Record sensitivity of each elements
	L = np.zeros([l1 * l2])
	for row_idx in range(l1):
		for col_idx in range(l2):
			L[row_idx * l2 + col_idx] = np.power(wb[row_idx, col_idx], 2) / (hessian_inv[row_idx, row_idx] + 10e-6)

	# Rank the sensitivity
	sen_rank = np.argsort(L)

	n_prune = l1 * l2
	save_interval = n_prune / 20 # 5% as save gap
	print '[%s] Prune number: %d' % (datetime.now(), n_prune)
	mask = np.ones(wb.shape)
	for i in xrange(n_prune):

		prune_idx = sen_rank[i]
		prune_row_idx = prune_idx / l2
		prune_col_idx = prune_idx % l2
		delta_W = - wb[prune_row_idx, prune_col_idx] / (hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
		try:
			wb[:, prune_col_idx] += delta_W
		except Warning:
			print Warning
			break
		mask[prune_row_idx, prune_col_idx] = 0

		if i % save_interval == 0 and i / save_interval >= 4:
			CR = 100 - (i / save_interval) * 5
			wb = np.multiply(wb, mask)
			print('[%s] Now save pruned weights of CR %d' %(datetime.now(), CR))
			# Save pruned weights
			if not os.path.exists('%s/CR_%s' %(save_root, CR)):
				os.makedirs('%s/CR_%s' %(save_root, CR))
			
			if layer_type == 'F':
				np.save('%s/CR_%s/%s.weight' %(save_root, CR, layer_name), wb[0: -1, :].transpose())
				np.save('%s/CR_%s/%s.bias' %(save_root, CR, layer_name), wb[-1, :].transpose())
			elif layer_type == 'C':
				kernel = fold_weights(wb[0 :-1, :], kernel_shape)
				bias = wb[-1, :]
				np.save('%s/CR_%s/%s.weight' %(save_root, CR, layer_name), kernel)
				np.save('%s/CR_%s/%s.bias' %(save_root, CR, layer_name), bias)
			elif layer_type == 'R':
				kernel = fold_weights(wb, kernel_shape)
				np.save('%s/CR_%s/%s.weight' %(save_root, CR, layer_name), kernel)
			if CR == 5:
				break