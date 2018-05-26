"""
This code generate Hessian inverse for AlexNet
"""

import torch
from hessian_utils import generate_hessian, generate_hessian_inv_Woodbury
from models.alexnet_layer_input import AlexNet

import os
from datetime import datetime
from numpy.linalg import inv, pinv
import numpy as np

import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Detect whether there is GPU.
use_cuda = torch.cuda.is_available()
# ------------------------------------User Configuration------------------------------
# If you meet the error of "Tensor in different GPUs", please set the following line
# in both this .py and hessian_utils.py to force them to share the same GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Parameters and path to specify
n_hessian_batch_used = 50000 # The number of batches used in generating Hessian inverse matrix
hessian_batch_size = 2 # Batch size of generating Hessian
stride_factor = 3 # Larger stride will leads to less memory used
hessian_inv_save_root = './AlexNet/hessian_inv_100k'
pretrain_model_path = './AlexNet/alexnet-owt-4df8aa71.pth'
if not os.path.exists(hessian_inv_save_root):
    os.makedirs(hessian_inv_save_root)
hessian_save_root = './AlexNet/hessian_100k'
if not os.path.exists(hessian_save_root):
    os.makedirs(hessian_save_root)
use_Woodbury = True # Whether to use Woodbury
use_Woodbury_tfbackend = True # If you use Woodbury, whether to use tf backend to calculate Woodbury
# ------------------------------------User Configuration------------------------------
# Load pretrain model
pretrain = torch.load(pretrain_model_path)
net = AlexNet()
net.load_state_dict(pretrain)
# Layer name of AlexNet
layer_name_list = [
    'features.0',
    'features.3',
    'features.6',
    'features.8',
    'features.10',
    'classifier.1',
    'classifier.4',
    'classifier.6',
]

# Generate Hessian
# traindir = '/home/shangyu/imagenet-train'
traindir = '/remote-imagenet/train' # Specify your imagenet train folder

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
    print ('[%s] %s' %(datetime.now(), layer_name))
    # if os.path.exists('%s/%s.npy' %(hessian_inv_save_root, layer_name)):
    #     continue
    # Generate Hessian
    if layer_name.startswith('features'):
        use_Woodbury = False
        if not use_Woodbury:
            hessian = generate_hessian(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'C', \
                        n_batch_used = n_hessian_batch_used, 
                        batch_size = hessian_batch_size,
                        stride_factor = stride_factor,
                        use_cuda = use_cuda)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'C', \
                        n_batch_used = n_hessian_batch_used, 
                        batch_size = hessian_batch_size,
                        stride_factor = stride_factor,
                        use_tf_backend = use_Woodbury_tfbackend,
                        use_cuda = use_cuda)
    elif layer_name.startswith('classifier'):
        use_Woodbury = True
        if not use_Woodbury:
            hessian = generate_hessian(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'F', \
                        n_batch_used = n_hessian_batch_used, 
                        batch_size = hessian_batch_size,
                        stride_factor = stride_factor,
                        use_cuda = use_cuda)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net = net, trainloader = hessian_loader, \
                        layer_name = layer_name, layer_type = 'F', \
                        n_batch_used = n_hessian_batch_used, 
                        batch_size = hessian_batch_size,
                        use_tf_backend = use_Woodbury_tfbackend,
                        stride_factor = stride_factor,
                        use_tf_backend = use_Woodbury_tfbackend,
                        use_cuda = use_cuda)
    
    if not use_Woodbury:
        np.save('%s/%s.npy' %(hessian_save_root, layer_name), hessian)
        # Inverse Hessian
        try:
            hessian_inv = inv(hessian)
        except Exception as err:
            print err
            hessian_inv = pinv(hessian)
    # Save hessian inverse 
    np.save('%s/%s.npy' %(hessian_inv_save_root, layer_name), hessian_inv)