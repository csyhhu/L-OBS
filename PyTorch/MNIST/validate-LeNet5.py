""" 
This code validates the performance of AlexNet after L-OBS prunning
""" 

import torch
import torch.backends.cudnn as cudnn

from models.LeNet5 import LeNet5
from utils.dataset import get_dataloader
from utils.train import validate

import numpy as np

import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

use_cuda = torch.cuda.is_available()
# -------------------------------------------- User Config ------------------------------------
# Specify parameters path
model_name = 'LeNet5'
pruned_weight_root = './%s/pruned_weight' %(model_name)
pretrain_model_path = './%s/%s_pretrain.pth' %(model_name, model_name)
n_validate_batch = -1 # Number of batches used for validation
validate_batch_size = 100 # Batch size of validation
prune_ratio = {
    'conv1': 60,
    'conv2': 60,
    'fc1': 5,
    'fc2': 30
}
# -------------------------------------------- User Config ------------------------------------
net = LeNet5()
net.load_state_dict(torch.load(pretrain_model_path))
param = net.state_dict()
total_nnz = 0
total_nelements = 0

for layer_name, CR in prune_ratio.items():

    if CR != 100:
        pruned_weight = np.load('%s/CR_%d/%s.weight.npy' %(pruned_weight_root, CR, layer_name))
        pruned_bias = np.load('%s/CR_%d/%s.bias.npy' %(pruned_weight_root, CR, layer_name))

        # Calculate sparsity
        total_nnz += np.count_nonzero(pruned_weight)
        total_nnz += np.count_nonzero(pruned_bias) 
        total_nelements += pruned_weight.size
        # total_nelements += pruned_bias.size

        param['%s.weight' %layer_name] = torch.FloatTensor(pruned_weight)
        # param['%s.bias' %layer_name] = torch.FloatTensor(pruned_bias)

        print(pruned_weight.size)

overall_CR = float(total_nnz) / float(total_nelements)
print ('Overall compression rate (nnz/total): %f' %overall_CR)
net.load_state_dict(param)
'''
print ('[%s] Begin adjust finish. Now saving parameters' %(datetime.now()))
adjust_mean_var(net, train_loader, None)
print ('[%s] Adjust finish. Now saving parameters' %(datetime.now()))
'''
# Load validation dataset
val_loader = get_dataloader('MNIST', 'test', validate_batch_size)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

validate(net, val_loader, None, None, n_validate_batch, use_cuda)