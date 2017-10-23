"""
This code
"""

import numpy as np
from datetime import datetime
import os


def fold_weights(w_matrix, hei, wid, in_channels):

	for i in range(w_matrix.shape[0]):
		if i == 0:
			real_w = w_matrix[i].reshape([hei, wid, in_channels, 1])
		else:
			real_w = np.concatenate((real_w, w_matrix[i].reshape([hei, wid, in_channels, 1])), axis=3)

	return real_w


def prune_weights_res(weights, hessian_inverse, CR):

	n_hidden_1 = int(weights.shape[0] * weights.shape[1] * weights.shape[2])
	n_hidden_2 = int(weights.shape[3])

	gate_w = np.ones([n_hidden_2, n_hidden_1])

	diag_hess_inv = np.add(np.diag(hessian_inverse), 1e-7)

	# Get each 'row' of kernels and flatten it
	for i in range(n_hidden_2):
		row_kernel = weights[:, :, :, i].reshape(-1)

		# Pack the unfolded weights, which reshape the kernels into a two-dimension array
		if i == 0:
			unfolded_weights = row_kernel
		else:
			unfolded_weights = np.vstack([unfolded_weights, row_kernel])

		# Get sensitivity matrix, arranged as unfolded weights
		if i == 0:
			sensitivity = (row_kernel ** 2 / diag_hess_inv)
		else:
			sensitivity = np.hstack([sensitivity, (row_kernel ** 2 / diag_hess_inv)])


	sorted_index = np.argsort(sensitivity)  # Sort from small to big

	# Begin pruning
	n_total = int(n_hidden_1 * n_hidden_2)
	n_total_prune = int(n_hidden_1 * n_hidden_2 * (1-CR))
	for i in range(n_total_prune):
		prune_index = sorted_index[i]

		x_index = prune_index / n_hidden_1  # next layer num  0----n_hidden_2
		y_index = prune_index % n_hidden_1  # this layer num  0----n_hidden_1

		delta_w = (-unfolded_weights[x_index][y_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[
			y_index]
		gate_w[x_index][y_index] = 0
		unfolded_weights[x_index] = unfolded_weights[x_index] + delta_w.T

		# Watch info
		if i % n_total == 0 and i != 0:
			CR = int(100 - (i / n_total) * 5)
			print '[%s] Now prune to CR: %d' % (datetime.now(), CR)

	unfolded_weights = unfolded_weights * gate_w
	# pruned_weights = unfolded_weights.reshape([weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]])
	pruned_weights = fold_weights(w_matrix=unfolded_weights, hei=weights.shape[0], wid=weights.shape[1],
								  in_channels=weights.shape[2])

	if not os.path.exists('pruned_weights/%s/' % layer_name):
		os.mkdir('pruned_weights/%s/' % layer_name)
	np.save('pruned_weights/%s/%d.npy' % (layer_name), pruned_weights)


def prune_weights_conv(weights, biases, hessian_inverse, CR):
	n_hidden_1 = int(weights.shape[0] * weights.shape[1] * weights.shape[2])
	n_hidden_2 = int(weights.shape[3])

	gate_w = np.ones([n_hidden_2, n_hidden_1])
	gate_b = np.ones([n_hidden_2])

	diag_hess_inv = np.add(np.diag(hessian_inverse), 1e-7)

	# Get each 'row' of kernels and flatten it
	for i in range(n_hidden_2):
		row_kernel = weights[:, :, :, i].reshape(-1)

		# Pack the unfolded weights, which reshape the kernels into a two-dimension array
		if i == 0:
			unfolded_weights = row_kernel
		else:
			unfolded_weights = np.vstack([unfolded_weights, row_kernel])

		# Get sensitivity matrix, arranged as unfolded weights
		if i == 0:
			sensitivity = np.hstack([row_kernel, np.array(biases)[i].reshape(-1)]) ** 2 / diag_hess_inv
		else:
			sensitivity = np.hstack(
				(sensitivity, (np.hstack((row_kernel, np.array(biases)[i].reshape(-1))) ** 2) / diag_hess_inv))

	sorted_index = np.argsort(sensitivity)  # Sort from small to big

	# Begin pruning
	n_total = int(n_hidden_1 * n_hidden_2)
	n_total_prune = int(n_hidden_1 * n_hidden_2 * (1 - CR))
	for i in range(n_total_prune):
		prune_index = sorted_index[i]
		x_index = prune_index / (n_hidden_1 + 1)  # next layer num  0----n_hidden_2
		y_index = prune_index % (n_hidden_1 + 1)  # this layer num  0----n_hidden_1

		if y_index == n_hidden_1:  # b
			delta_w = (-biases[x_index] / (hessian_inverse[y_index][y_index])) * hessian_inverse.T[y_index]
			gate_b[x_index] = 0
		else:
			delta_w = (-unfolded_weights[x_index][y_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[
				y_index]
			gate_w[x_index][y_index] = 0
			unfolded_weights[x_index] = unfolded_weights[x_index] + delta_w[0: -1].T

		biases[x_index] = biases[x_index] + delta_w[-1]

		# Watch info
		if i % n_total == 0 and i != 0:
			CR = int(100 - (i / n_total) * 5)
			print '[%s] Now prune to CR: %d' % (datetime.now(), CR)

	unfolded_weights = unfolded_weights * gate_w
	biases = biases * gate_b
	# pruned_weights = unfolded_weights.reshape([weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]])
	pruned_weights = fold_weights(w_matrix=unfolded_weights, hei=weights.shape[0], wid=weights.shape[1],
								  in_channels=weights.shape[2])

	if not os.path.exists('pruned_weights/%s/' % layer_name):
		os.mkdir('pruned_weights/%s/' % layer_name)
	np.save('pruned_weights/%s/weights.npy' % (layer_name), pruned_weights)
	np.save('pruned_weights/%s/biases.npy'% (layer_name), biases)


def prune_weights_fc(weights, biases, hessian_inverse, CR):

	n_hidden_1 = int(weights.shape[0])
	n_hidden_2 = int(weights.shape[1])

	gate_w = np.ones([n_hidden_1, n_hidden_2])
	gate_b = np.ones([n_hidden_2])

	sensitivity = np.array([])

	for i in range(n_hidden_2):
		sensitivity = np.hstack(
			(sensitivity, 0.5 * (np.hstack((weights.T[i], biases[i])) ** 2) / np.diag(hessian_inverse)))

	sorted_index = np.argsort(sensitivity)  # Sort from small to big

	# Begin pruning
	n_total = int(n_hidden_1 * n_hidden_2)
	n_total_prune = int(n_hidden_1 * n_hidden_2 * (1 - CR))
	for i in range(n_total_prune):
		prune_index = sorted_index[i]
		x_index = prune_index / (n_hidden_1 + 1)  # next layer num  0----n_hidden_2
		y_index = prune_index % (n_hidden_1 + 1)  # this layer num  0----n_hidden_1

		if y_index == n_hidden_1:  # b
			delta_w = (-biases[x_index] / (hessian_inverse[y_index][y_index])) * hessian_inverse.T[y_index]
			gate_b[x_index] = 0
		else:
			delta_w = (-weights[x_index][y_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[
				y_index]
			gate_w[x_index][y_index] = 0
			weights[x_index] = weights[x_index] + delta_w[0: -1].T

		biases[x_index] = biases[x_index] + delta_w[-1]

		# Watch info
		if i % n_total == 0 and i != 0:
			CR = int(100 - (i / n_total) * 5)
			print '[%s] Now prune to CR: %d' % (datetime.now(), CR)

	weights = weights * gate_w
	biases = biases * gate_b

	if not os.path.exists('pruned_weights/%s/' % layer_name):
		os.mkdir('pruned_weights/%s/' % layer_name)
	np.save('pruned_weights/%s/weights.npy' % (layer_name, CR), weights)
	np.save('pruned_weights/%s/biases.npy' % (layer_name, CR), biases)


# Load all the parameters
paramters = np.load('res50.npy').item()
layer_name_list = np.load('res50_layer_name.npy')

for layer_name in layer_name_list:

	hessian_inverse = np.load('hessian/%s.npy' %layer_name)

	if layer_name.startswith('conv'):
		prune_weights_conv(paramters[layer_name]['weights'], paramters[layer_name]['biases'], hessian_inverse, CR=40)

	elif layer_name.startswith('fc'):
		prune_weights_conv(paramters[layer_name]['weights'], paramters[layer_name]['biases'], hessian_inverse, CR=40)

	elif layer_name.startswith('res'):
		prune_weights_conv(paramters[layer_name]['weights'], hessian_inverse, CR=40)