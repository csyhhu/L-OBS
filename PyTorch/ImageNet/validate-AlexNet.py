""" 
This code validates the performance of AlexNet after L-OBS prunning
""" 

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.alexnet import AlexNet
from utils import validate, adjust_mean_var

import numpy as np 
from datetime import datetime

use_cuda = torch.cuda.is_available()
# -------------------------------------------- User Config ------------------------------------
# Specify parameters path
pruned_weight_root = './AlexNet/pruned_weight'
pretrain_model_path = './AlexNet/alexnet-owt-4df8aa71.pth'
n_validate_batch = 100 # Number of batches used for validation
validate_batch_size = 50 # Batch size of validation
prune_ratio = {
    'features.0': 80,
    'features.3': 35,
    'features.6': 35,
    'features.8': 35, 
    'features.10': 35,
    'classifier.1': 10,
    'classifier.4': 10,
    'classifier.6': 25
}
# -------------------------------------------- User Config ------------------------------------
net = AlexNet()
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
        total_nelements += pruned_bias.size

        param['%s.weight' %layer_name] = torch.FloatTensor(pruned_weight)
        param['%s.bias' %layer_name] = torch.FloatTensor(pruned_bias)

overall_CR = float(total_nnz) / float(total_nelements)
print ('Overall compression rate (nnz/total): %f' %overall_CR)
net.load_state_dict(param)

'''
print ('[%s] Begin adjust finish. Now saving parameters' %(datetime.now()))
adjust_mean_var(net, train_loader, None)
print ('[%s] Adjust finish. Now saving parameters' %(datetime.now()))
'''
# Load validation dataset
valdir = '/home/shangyu/imagenet-val'

print('==> Preparing data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])),
	batch_size = validate_batch_size, shuffle=True)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

validate(net, val_loader, None, None, n_validate_batch, use_cuda)