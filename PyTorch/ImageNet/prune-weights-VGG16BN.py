""" 
This code prune weights in VGG16BN usin L-OBS
""" 
import os
import numpy as np
from datetime import datetime
import cPickle

from utils import unfold_kernel, fold_weights, get_error

import torch

# data path specify
hessian_inverse_root = './VGG16/hessian_inv'
save_root = './VGG16/pruned_weight'
if not os.path.exists(save_root):
	os.makedirs(save_root)
rank_root = './VGG16/sensitivity' # Specify your parameter rank file save root
if not os.path.exists(rank_root):
	os.makedirs(rank_root)

# Layer name of VGG16
layer_name_list = [
    'features.0',
    'features.3',
    'features.7',
    'features.10',
    'features.14',
    'features.17',
    'features.20',
    'features.24',
    'features.27',
    'features.30',
    'features.34',
    'features.37',
    'features.40',
    'classifier.0',
    'classifier.3',
    'classifier.6'
]

pretrain = torch.load('./VGG16/vgg16_bn-6c64b313.pth')

for layer_idx, layer_name in enumerate(layer_name_list):

	# if os.path.exists('%s/CR_5/%s.weight.npy' %(save_root, layer_name)):
	# 	continue

	print ('[%s] %s' %(datetime.now(), layer_name))
	# Specify layer type, C for convolution, F for fully-connected
	if layer_name.startswith('features'):
		layer_type = 'C'
	elif layer_name.startswith('classifier'):
		layer_type = 'F'

	if layer_type == 'C':
		kernel = pretrain['%s.weight' %layer_name].numpy()
		kernel_shape = kernel.shape # [64, 3, 11, 11]
		weight = unfold_kernel(kernel) # [364, 64]
		bias = pretrain['%s.bias' %layer_name].numpy() # [64, ]
		wb = np.concatenate([weight, bias.reshape(1, -1)], axis = 0) # [365, 64]
	elif layer_type == 'F':
		weight = pretrain['%s.weight' %layer_name].numpy()
		bias = pretrain['%s.bias' %layer_name].numpy()
		wb = np.hstack([weight, bias.reshape(-1, 1)]).transpose()

	hessian_inv = np.load('%s/%s.npy' %(hessian_inverse_root, layer_name))

	l1, l2 = wb.shape
	
	if os.path.exists('%s/%s.npy' %(rank_root, layer_name)):
		print ('Rank exist, loading.')
		sen_rank = np.load('%s/%s.npy' %(rank_root, layer_name))
	else:
		print ('Rank not exist, create one.')
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
		try:
			delta_W = - wb[prune_row_idx, prune_col_idx] / (hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
		except Warning:
			print ('Nan found, please change another Hessian inverse calculation method')
			break
		wb[:, prune_col_idx] += delta_W
		mask[prune_row_idx, prune_col_idx] = 0

		# wb = np.multiply(wb, mask)

		if i % save_interval == 0 and i / save_interval >= 4:
			wb = np.multiply(wb, mask)
			CR = 100 - (i / save_interval) * 5
			print ('Construct element-wise multiplication between weight and mask matrix graph')
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
	
	# cPickle.dump(layer_error, open('%s/%s.pkl' %(layer_error_save_root, layer_name), 'w'))
	# break