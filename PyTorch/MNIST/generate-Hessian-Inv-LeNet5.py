import torch

import os
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from datetime import datetime

from utils.dataset import get_dataloader
from utils.hessian_utils import generate_hessian, generate_hessian_inv_Woodbury
from models.LeNet5_layer_input import LeNet5

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ------------- Configuration --------------
use_cuda = torch.cuda.is_available()
dataset_name = 'MNIST'
model_name = 'LeNet5'
hessian_batch_size = 2
hessian_inv_save_root = './%s/hessian_inv' %model_name
if not os.path.exists(hessian_inv_save_root):
    os.makedirs(hessian_inv_save_root)
hessian_save_root = './%s/hessian' %model_name
if not os.path.exists(hessian_save_root):
    os.makedirs(hessian_save_root)
pretrain_model_path = './%s/%s_pretrain.pth' %(model_name, model_name)
use_Woodbury_tfbackend = False
# -----------------------------------------

##############
# Buil Model #
##############
net = LeNet5()
net.load_state_dict(torch.load(pretrain_model_path))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

################
# Load Dataset #
################
hessian_loader = get_dataloader(dataset_name=dataset_name, split='train', batch_size=hessian_batch_size)

############################
# Generate Hessian-Inverse #
############################
layer_name_list = ['conv1', 'conv2', 'fc1', 'fc2']
use_Woodbury_list = []
for layer_name in layer_name_list:

    if layer_name in use_Woodbury_list:
        use_Woodbury = True
    else:
        use_Woodbury = False
    print('[%s] %s. Method: %s' % (datetime.now(), layer_name, 'Woodbury' if use_Woodbury else 'Normal'))
    # Generate Hessian

    if 'fc' in layer_name:
        if not use_Woodbury:
            hessian = generate_hessian(net=net, trainloader=hessian_loader, \
                                       layer_name=layer_name, layer_type='F', \
                                       n_batch_used=-1,
                                       batch_size=hessian_batch_size)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net=net, trainloader=hessian_loader, \
                                                        layer_name=layer_name, layer_type='F', \
                                                        n_batch_used=100,
                                                        batch_size=hessian_batch_size,
                                                        use_tf_backend=use_Woodbury_tfbackend)
    else:
        if not use_Woodbury:
            hessian = generate_hessian(net=net, trainloader=hessian_loader, \
                                       layer_name=layer_name, layer_type='C', \
                                       n_batch_used=-1,
                                       batch_size=hessian_batch_size)
        else:
            hessian_inv = generate_hessian_inv_Woodbury(net=net, trainloader=hessian_loader, \
                                                        layer_name=layer_name, layer_type='C', \
                                                        n_batch_used=100,
                                                        batch_size=hessian_batch_size,
                                                        use_tf_backend=use_Woodbury_tfbackend)

    if not use_Woodbury:
        np.save('%s/%s.npy' % (hessian_save_root, layer_name), hessian)
        # Inverse Hessian
        try:
            hessian_inv = inv(hessian)
        except Exception:
            print (Exception)
            hessian_inv = pinv(hessian)
    # Save hessian inverse
    np.save('%s/%s.npy' % (hessian_inv_save_root, layer_name), hessian_inv)