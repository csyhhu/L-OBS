"""
This code generate Hessian inverse for VGG16 BN version
"""

import torch
from hessian_utils import generate_hessian, generate_hessian_inv_Woodbury
from models.resnet_layer_input import resnet18

import os
from datetime import datetime
from numpy.linalg import inv, pinv, LinAlgError
import numpy as np

import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Detect whether there is GPU.
use_cuda = torch.cuda.is_available()
# -------------------------------- User Config ----------------------------------------------
# If you meet the error of "Tensor in different GPUs", please set the following line
# in both this .py and hessian_utils.py to force them to share the same GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Parameters and path to specify
hessian_batch_size = 2
hessian_inv_save_root = './ResNet18/hessian_inv'
if not os.path.exists(hessian_inv_save_root):
    os.makedirs(hessian_inv_save_root)
hessian_save_root = './ResNet18/hessian'
if not os.path.exists(hessian_save_root):
    os.makedirs(hessian_save_root)
pretrain_model_path = './ResNet18/resnet18-5c106cde.pth'
use_Woodbury = True # Whether to use Woodbury
use_Woodbury_tfbackend = True # If you use Woodbury, whether to use tf backend to calculate Woodbury
traindir = '/home/shangyu/imagenet-train' # Specify your imagenet training data
# -------------------------------- User Config ----------------------------------------------
# Load pretrain model
pretrain = torch.load(pretrain_model_path)
net = resnet18()
net.load_state_dict(pretrain)

# Layer name of ResNet
layer_name_list = list()
layer_name_list.append('fc')

for layer_name in net.state_dict().keys():
    if 'conv1.weight' in layer_name or 'conv2.weight' in layer_name or 'conv3.weight' in layer_name \
        or 'downsample.0.weight' in layer_name:
        layer_name_list.append(layer_name[: -7])

# These list is only available for ResNet18
use_Woodbury_list = [
	'layer4.0.conv1',
	'layer4.0.conv2',
	'layer4.1.conv1',
	'layer4.1.conv2',
    'layer3.0.conv2'
]
# Generate Hessian

print('==> Preparing data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trainDataset = datasets.ImageFolder(traindir, transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	]))

hessian_loader = torch.utils.data.DataLoader(trainDataset, batch_size = hessian_batch_size, shuffle=True)
print ('[%s] Finish process data' % datetime.now())

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

for layer_name in layer_name_list:
    if layer_name in use_Woodbury_list:
        use_Woodbury = True
    else:
        use_Woodbury = False
    print ('[%s] %s. Method: %s' %(datetime.now(), layer_name, 'Woodbury' if use_Woodbury else 'Normal'))
    # Generate Hessian

    if layer_name == 'fc':
        if not use_Woodbury:
            hessian = generate_hessian(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'F', \
                        n_batch_used = 100, 
                        batch_size = hessian_batch_size)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net = net, trainloader = hessian_loader, \
                            layer_name = layer_name, layer_type = 'F', \
                            n_batch_used = 100, 
                            batch_size = hessian_batch_size,
                            use_tf_backend = use_Woodbury_tfbackend)
    else:
        if not use_Woodbury:
            hessian = generate_hessian(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'R', \
                        n_batch_used = 100, 
                        batch_size = hessian_batch_size)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net = net, trainloader = hessian_loader, \
                            layer_name = layer_name, layer_type = 'R', \
                            n_batch_used = 100, 
                            batch_size = hessian_batch_size,
                            use_tf_backend = use_Woodbury_tfbackend)
    
    if not use_Woodbury:
        np.save('%s/%s.npy' %(hessian_save_root, layer_name), hessian)
        # Inverse Hessian
        try:
            hessian_inv = inv(hessian)
        except Error:
            print Error
            hessian_inv = pinv(hessian)
    # Save hessian inverse 
    np.save('%s/%s.npy' %(hessian_inv_save_root, layer_name), hessian_inv)