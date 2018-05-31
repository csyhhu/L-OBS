""" 
This code validates the performance of ResNet after L-OBS prunning
""" 

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.resnet import resnet18
from utils import validate, adjust_mean_var

import numpy as np 
import os
from datetime import datetime

use_cuda = torch.cuda.is_available()
# -------------------------------------------- User Config ------------------------------------
# Specify parameters path
traindir = '/home/shangyu/imagenet-train'
valdir = '/home/shangyu/imagenet-val'
pruned_weight_root = './ResNet18/pruned_weight'
pruned_parameter_root = './ResNet18/pruned_param'
if not os.path.exists(pruned_parameter_root):
    os.makedirs(pruned_parameter_root)
pretrain_model_path = './ResNet18/resnet18-5c106cde.pth'
n_validate_batch = 100 # Number of batches used for validation
validate_batch_size = 50 # Batch size of validation
adjust_batch_size = 128
n_adjust_batch = 500
CR = 60 # Compression rate for all layers
# -------------------------------------------- User Config ------------------------------------
net = resnet18()
# net.load_state_dict(torch.load(pretrain_model_path))
# param = net.state_dict()
param = torch.load(pretrain_model_path)
total_nnz = 0
total_nelements = 0
n_weight_used = 0
n_total_weight = len(os.listdir('%s/CR_%d' %(pruned_weight_root, CR)))

for layer_name in param.keys():

    if not os.path.exists('%s/CR_%d/%s.npy' %(pruned_weight_root, CR, layer_name)):
        continue

    pruned_weight = np.load('%s/CR_%d/%s.npy' %(pruned_weight_root, CR, layer_name))
    # print pruned_weight
    # raw_input()
    # Calculate sparsity
    this_sparsity = np.count_nonzero(pruned_weight)
    this_total = pruned_weight.size
    print ('%s CR: %f' %(layer_name, float(this_sparsity)/float(this_total)))
    total_nnz += this_sparsity
    total_nelements += this_total

    param['%s' %layer_name] = torch.FloatTensor(pruned_weight)
    n_weight_used += 1

# assert(n_weight_used == n_total_weight)
print ('Prune weights used: %d/%d' %(n_weight_used, n_total_weight))
overall_CR = float(total_nnz) / float(total_nelements)
print ('Overall compression rate (nnz/total): %f' %overall_CR)
net.load_state_dict(param)
torch.save(param, open('%s/CR-%d.pth' %(pruned_parameter_root, CR), 'w'))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# Load training dataset for mean/var adjust
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trainDataset = datasets.ImageFolder(traindir, transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	]))

train_loader = torch.utils.data.DataLoader(trainDataset, batch_size = adjust_batch_size, shuffle=True)
print ('[%s] Begin adjust finish' %(datetime.now()))
adjust_mean_var(net, train_loader, None, n_adjust_batch, use_cuda)
print ('[%s] Adjust finish. Now saving parameters' %(datetime.now()))

# Load validation dataset
print('==> Preparing data..')
val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])),
	batch_size = validate_batch_size, shuffle=True)

validate(net, val_loader, None, None, n_validate_batch, use_cuda)