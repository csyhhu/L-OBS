"""
This file contains some utility functions for using ImageNet dataset and L-OBS

Author: Chen Shangyu (schen025@e.ntu.edu.sg)

"""
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim

import collections
from datetime import datetime

import numpy as np

import tensorflow as tf

import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pickle

use_cuda = torch.cuda.is_available()


def get_error(theta_B, hessian, theta_0):
	"""
	Calculate \delta \theta^T H \delta \theta
	:param theta_B:
	:param hessian:
	:param theta_0:
	:param alpha:
	:param sigma:
	:return:
	"""

	delta = theta_B - theta_0
	error = np.trace(np.dot(np.dot(delta.T, hessian), delta))

	return error


def unfold_kernel(kernel):
	"""
	In pytorch format, kernel is stored as [out_channel, in_channel, height, width]
	Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
	:param kernel: numpy ndarray
	:return:
	"""
	k_shape = kernel.shape
	weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
	for i in range(k_shape[0]):
		weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

	return weight


def fold_weights(weights, kernel_shape):
	"""
	In pytorch format, kernel is stored as [out_channel, in_channel, width, height]
	Fold weights into a 4-dimensional tensor as [out_channel, in_channel, width, height]
	:param weights:
	:param kernel_shape:
	:return:
	"""
	kernel = np.zeros(shape=kernel_shape)
	for i in range(kernel_shape[0]):
		kernel[i,:,:,:] = weights[:, i].reshape([kernel_shape[1], kernel_shape[2], kernel_shape[3]])

	return kernel



def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def validate(model, val_loader, val_record, train_record, n_batch_used = 100, use_cuda = True):

	monitor_freq = int(n_batch_used / 5)

	# batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	criterion = nn.CrossEntropyLoss()

	# switch to evaluate mode
	model.eval()

	# end = time.time()
	for i, (input, target) in enumerate(val_loader):
		if use_cuda:
			target = target.cuda()
			input = input.cuda()
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		if i % monitor_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   i, len(val_loader), loss=losses,
				   top1=top1, top5=top5))
		
		if i == n_batch_used:
			break

	print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
		  .format(top1=top1, top5=top5))
	if val_record != None:
		val_record.write('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
			.format(top1=top1, top5=top5))
	if train_record != None:
		train_record.write('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
			.format(top1=top1, top5=top5))
	
	model.train()

	return top1.avg, top5.avg


def adjust_mean_var(net, train_loader, train_file, n_batch_used = 500, use_cuda = True):

	monitor_freq = int(n_batch_used / 5)

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	criterion = nn.CrossEntropyLoss()

	model.train()

	# end = time.time()
	for i, (input, target) in enumerate(val_loader):
		if use_cuda:
			target = target.cuda()
			input = input.cuda()
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)

		loss = criterion(output, target_var)
		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		if (i+1) % monitor_freq == 0:
			print('Train: [{0}/{1}]\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   i, n_batch_used, loss=losses,
				   top1=top1, top5=top5))
			if train_file != None:
				train_file.write('[%d/%d] Loss: %f, Prec@1: %f, Prec@5: %f\n' %\
					(i, n_batch_used, losses.avg, top1.avg, top5.avg))
		
		if (i+1) == n_batch_used:
			break 


def create_prune_graph(input_dimension, output_dimension):
	pruned_weight_holder = tf.placeholder(tf.float32, shape=None)
	hessian_inv_diag_holder = tf.placeholder(tf.float32, shape=None)
	hessian_inv_holder = tf.placeholder(tf.float32, shape=[input_dimension, input_dimension])
	prune_row_idx_holder = tf.placeholder(tf.float32, shape=None)
	mask_holder = tf.placeholder(tf.float32, shape=[input_dimension, output_dimension])
	wb_holder = tf.placeholder(tf.float32, shape=[input_dimension, output_dimension])
	
	selection_q = tf.one_hot(indices = prune_row_idx_holder, depth = input_dimension)
	get_sparse_wb_op = -pruned_weight_holder / (hessian_inv_diag_holder + 10e-6) \
		* tf.matmul(a = hessian_inv_holder, b = selection_q) + wb_holder
	return pruned_weight_holder, hessian_inv_diag_holder, hessian_inv_holder, prune_row_idx_holder,\
			mask_holder, wb_holder, get_sparse_wb_op
	

def create_sparse_mul_graph(input_dimension, output_dimension):
	""" 
	This function perform element-wise multiplication between weights matrix and mask matrix
	by tensorflow backend to speed up
	args:
		input_dimension: first dimension of weights (mask) matrix
		output_dimension: second dimension of weights (mask) matrix
	Output:
		mask_holder: tf holder for mask matrix
		wb_holder: tf holder for weight matrix
		get_sparse_wb_op: tf op for generating sparse wb
	""" 
	mask_holder = tf.placeholder(tf.float32, shape=[input_dimension, output_dimension])
	wb_holder = tf.placeholder(tf.float32, shape=[input_dimension, output_dimension])

	get_sparse_wb_op = tf.multiply(wb_holder, mask_holder)

	return mask_holder, wb_holder, get_sparse_wb_op


def generate_layer_list(param):
	pass