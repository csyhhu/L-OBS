""" 
This file contains some utility functions to calculate hessian matrix and its inverse.

Author: Chen Shangyu (schen025@e.ntu.edu.sg)
""" 

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from datetime import datetime

import tensorflow as tf
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Construct hessian computing graph for res layer (conv layer without bias)
def create_res_hessian_computing_tf_graph(input_shape, layer_kernel, layer_stride):
    """ 
    This function create the TensorFlow graph for computing hessian matrix for res layer.
    Step 1: It first extract image patches using tf.extract_image_patches.
    Step 2: Then calculate the hessian matrix by outer product.

    Args:
        input_shape: the dimension of input
        layer_kernel: kernel size of the layer
        layer_stride: stride of the layer
    Output:
        input_holder: TensorFlow placeholder for layer input
        get_hessian_op: A TensorFlow operator to calculate hessian matrix

    """ 
    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    patches = tf.extract_image_patches(images = input_holder,
                                       ksizes = [1,layer_kernel, layer_kernel,1],
                                       strides = [1, layer_stride, layer_stride, 1],
                                       rates = [1, 1, 1, 1],
                                       padding = 'SAME')
    print 'Patches shape: %s' %patches.get_shape()
    a = tf.expand_dims(patches, axis=-1)
    b = tf.expand_dims(patches, axis=3)
    outprod = tf.multiply(a, b)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    print 'Hessian shape: %s' % get_hessian_op.get_shape()
    return input_holder, get_hessian_op


# Construct hessian computing graph for fc layer
def create_fc_hessian_computing_tf_graph(input_shape):
    """ 
    This function create the TensorFlow graph for computing hessian matrix for fully-connected layer.
    Compared with create_res_hessian_computing_tf_graph, it does not need to extract patches.
    """ 

    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    a = tf.expand_dims(input_holder, axis=-1)
    # Appending extra one for bias term
    vect_w_b = tf.concat([a, tf.ones([tf.shape(a)[0], 1, 1])], axis=1)
    outprod = tf.matmul(vect_w_b, vect_w_b, transpose_b=True)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=0)
    print 'Hessian shape: %s' % get_hessian_op.get_shape()
    return input_holder, get_hessian_op


# Construct hessian computing graph
def create_conv_hessian_computing_tf_graph(input_shape, layer_kernel, layer_stride):
    """ 
    This function create the TensorFlow graph for computing hessian matrix for fully-connected layer.
    Compared with create_res_hessian_computing_tf_graph, it append extract one for bias term.
    """ 
    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    patches = tf.extract_image_patches(images = input_holder,
                                       ksizes = [1,layer_kernel, layer_kernel,1],
                                       strides = [1, layer_stride, layer_stride, 1],
                                       rates = [1, 1, 1, 1],
                                       padding = 'SAME')
    print 'Patches shape: %s' %patches.get_shape()
    vect_w_b = tf.concat([patches, tf.ones([tf.shape(patches)[0], \
                tf.shape(patches)[1], tf.shape(patches)[2], 1])], axis=3)
    a = tf.expand_dims(vect_w_b, axis=-1)
    b = tf.expand_dims(vect_w_b, axis=3)
    outprod = tf.multiply(a, b)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    print 'Hessian shape: %s' % get_hessian_op.get_shape()
    return input_holder, get_hessian_op


# Construct hessian inverse computing graph for Woodbury
def create_Woodbury_hessian_inv_graph(input_shape, dataset_size):
    """
    This function create the hessian inverse calculation graph using Woodbury method.
    """ 

    hessian_inv_holder = tf.placeholder(dtype=tf.float32, shape=[input_shape, input_shape])
    input_holder = tf.placeholder(dtype=tf.float32, shape=[1, input_shape])
    # [1, 4097] [4097, 4097] [4097, 1]
    denominator = dataset_size + \
        tf.matmul(a = tf.matmul(a = input_holder, b = hessian_inv_holder), b = input_holder, transpose_b=True)
    # ([4097, 4097] [4097, 1]) ([1, 4097] [4097, 4097])
    numerator = tf.matmul(a = tf.matmul(a = hessian_inv_holder, b = input_holder, transpose_b=True), \
                            b = tf.matmul(a = input_holder, b = hessian_inv_holder))
    hessian_inv_op = hessian_inv_holder - numerator * (1.00 / denominator)

    return hessian_inv_holder, input_holder, hessian_inv_op


def generate_hessian(net, trainloader, layer_name, layer_type, \
    n_batch_used = 100, batch_size = 2, stride_factor = 3 ,use_cuda = True):
    """ 
    This function generate hessian matrix for a given layer. Basically, what it does is:
    Step 1: Extract layer input using PyTorch interface
    Step 2: For convolution, res layer, extract patches using TensorFlow function 
    Step 3: Calculate hessian
    Args:
        net: PyTorch model
        trainloader: PyTorch dataloader
        layer_name:
        layer_type: 'C' for Convolution (with bias), 'R' for res layer (without bias),
                    'F' for Fully-Connected (with bias). I am sure you will know why the bias term
                    is emphasized here as you are clever.
        n_batch_used: number of batches used to generate hessian.
        batch_size: Batch size. Because hessian calculation graph is quite huge. A small (like 2) number
                    of batch size if recommended here.
        stride_factor: Due to the same reason mentioned above, bigger stride results in fewer extracted
                    image patches (think about how convolution works).  stride_factor is multiplied by 
                    actual stride in latter use. Therefore when stride_factor == 1, it extract patches in
                    original way. However, it may results in some GPU/CPU memory troubles. If you meet such,
                    you can increase the stride factor here.
        use_cuda: whether you can use cuda or not.
    Output:
        Hessian matrix
    """ 

    freq_moniter = (n_batch_used * batch_size) / 50 # Total 50 times of printing information

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    net.eval()
    for batch_idx, (inputs, _) in enumerate(trainloader):
        
        if use_cuda:
            inputs = inputs.cuda()
        
        net(Variable(inputs, volatile=True))

        layer_input = net.module.layer_input[layer_name]

        # In the begining, construct hessian graph
        if batch_idx == 0:
            print '[%s] Now construct generate hessian op for layer %s' %(datetime.now(), layer_name)
            # res layer
            if layer_type == 'R':
                # Because PyTorch's data format (N,C,W,H) is different from tensorflow (N,W,H,C)
                # layer input should be permuted to fit tensorflow
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_res_hessian_computing_tf_graph(layer_input_np.shape, 
                                                        net.module.layer_kernel[layer_name], 
                                                        net.module.layer_stride[layer_name] * stride_factor)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print 'Hessian shape: %d' %hessian_shape
                weight_shape = net.state_dict()['module.%s.weight' %layer_name].size()
                # print ('Kernel shape: %s' %weight_shape)
                # print weight_shape
                kernel_unfold_shape = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
                print 'Kernel unfold shape: %d' %kernel_unfold_shape
                assert(hessian_shape == kernel_unfold_shape)
            # linear layer
            elif layer_type == 'F':
                layer_input_np = layer_input.cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_fc_hessian_computing_tf_graph(layer_input_np.shape)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print 'Hessian shape: %d' % hessian_shape
                weight_shape = net.state_dict()['module.%s.weight' % layer_name].size()
                print 'Weights shape: %d' % weight_shape[1]
                assert(hessian_shape == weight_shape[1] + 1) # +1 because of bias 
            elif layer_type == 'C':
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_conv_hessian_computing_tf_graph(layer_input_np.shape, 
                                                        net.module.layer_kernel[layer_name], 
                                                        net.module.layer_stride[layer_name] * stride_factor)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print 'Hessian shape: %d' %hessian_shape
                weight_shape = net.state_dict()['module.%s.weight' %layer_name].size()
                # print ('Kernel shape: %s' %weight_shape)
                # print weight_shape
                kernel_unfold_shape = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
                print 'Kernel unfold shape: %d' %kernel_unfold_shape
                assert(hessian_shape == kernel_unfold_shape + 1)

            print '[%s] %s Graph build complete.'  % (datetime.now(), layer_name)
    
        # Initialization finish, begin to calculate
        if layer_type == 'C' or layer_type == 'R':
            this_layer_input = layer_input.permute(0, 2, 3, 1).cpu().numpy()
        elif layer_type == 'F':
            this_layer_input = layer_input.cpu().numpy()

        this_hessian = sess.run(generate_hessian_op,
                                feed_dict={layer_input_holder: this_layer_input})

        if batch_idx == 0:
            layer_hessian = this_hessian
        else:
            layer_hessian += this_hessian

        if batch_idx % freq_moniter == 0:
            print '[%s] Now finish image No. %d / %d' \
                %(datetime.now(), batch_idx * batch_size, n_batch_used * batch_size)
    
        if batch_idx == n_batch_used:
            break

    # net.train()

    return (1.0 / n_batch_used) * layer_hessian


def generate_hessian_inv_Woodbury(net, trainloader, layer_name, layer_type, \
    n_batch_used = 100, batch_size = 2, stride_factor = 3 , use_tf_backend = True, use_cuda = True):
    """ 
    This function calculated Hessian inverse matrix by Woodbury matrix identity.
    Args:
        Please find the same parameters explanations above.
        use_tf_backend: A TensorFlow wrapper is used to accelerate the process. True for using such wrapper.
    """ 
    hessian_inverse = None
    dataset_size = 0
    freq_moniter = (n_batch_used * batch_size) / 50 # Total 50 times of printing information

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    net.eval()
    for batch_idx, (inputs, _) in enumerate(trainloader):

        if use_cuda:
            inputs = inputs.cuda()
    
        net(Variable(inputs, volatile=True))

        layer_input = net.module.layer_input[layer_name]

        # Construct tf op for convolution and res layer
        if batch_idx == 0:
            if layer_type == 'C' or layer_type == 'R':
                print '[%s] Now construct patches extraction op for layer %s' %(datetime.now(), layer_name)
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_kernel = net.module.layer_kernel[layer_name]
                layer_stride = net.module.layer_stride[layer_name] * stride_factor
                layer_input_holder = tf.placeholder(dtype=tf.float32, shape=layer_input_np.shape)

                get_patches_op = \
                    tf.extract_image_patches(images = layer_input_holder,
                                       ksizes = [1, layer_kernel, layer_kernel,1],
                                       strides = [1, layer_stride, layer_stride, 1],
                                       rates = [1, 1, 1, 1],
                                       padding = 'SAME')
                # For a convolution input, extracted pathes would be: [1, 9, 9, 2304]
                dataset_size = n_batch_used * int(get_patches_op.get_shape()[0]) * \
                    int(get_patches_op.get_shape()[1]) * int(get_patches_op.get_shape()[2])
                input_dimension = get_patches_op.get_shape()[3]
                if layer_type == 'C':
                    # In convolution layer, input dimension should be added one for bias term
                    hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                    if use_tf_backend:
                        print ('You choose tf backend to calculate Woodbury, constructing your graph.')
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = \
                            create_Woodbury_hessian_inv_graph(input_dimension + 1, dataset_size)
                else:
                    hessian_inverse = 1000000 * np.eye(input_dimension)
                    if use_tf_backend:
                        print ('You choose tf backend to calculate Woodbury, constructing your graph.')
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = \
                            create_Woodbury_hessian_inv_graph(input_dimension, dataset_size)
            else:
                layer_input_np = layer_input.cpu().numpy()
                input_dimension = layer_input_np.shape[1]
                dataset_size = n_batch_used * batch_size
                hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                if use_tf_backend:
                    print ('You choose tf backend to calculate Woodbury, constructing your graph.')
                    hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = \
                        create_Woodbury_hessian_inv_graph(input_dimension + 1, dataset_size)
            
            print '[%s] dataset: %d, input dimension: %d' %(datetime.now(), dataset_size, input_dimension)
        
        # Begin process
        if layer_type == 'F':
            this_layer_input = layer_input.cpu().numpy() # [2, 4096]
            for i in range(this_layer_input.shape[0]):
                this_input = this_layer_input[i]
                # print this_input.shape
                # print np.array([1.0]).shape
                wb = np.concatenate([this_input.reshape(1,-1), np.array([1.0]).reshape(1,-1)], axis = 1) # [1, 4097]
                if use_tf_backend:
                    hessian_inverse = sess.run(Woodbury_hessian_inv_op, feed_dict={
                        hessian_inv_holder: hessian_inverse,
                        input_holder: wb
                    })
                else:
                    # [1, 4097] [4097, 4097] [4097, 1]
                    denominator = dataset_size + np.dot(np.dot(wb,hessian_inverse), wb.T) 
                    # [4097, 4097] [4097, 1] [1, 4097] [4097, 4097]
                    numerator = np.dot(np.dot(hessian_inverse, wb.T), np.dot(wb,hessian_inverse))
                    hessian_inverse = hessian_inverse - numerator * (1.0 / denominator)
        
        elif layer_type == 'C' or layer_type == 'R':
            this_layer_input = layer_input.permute(0, 2, 3, 1).cpu().numpy()
            this_patch = sess.run(get_patches_op, feed_dict={layer_input_holder: this_layer_input})

            for i in range(this_patch.shape[0]):
                for j in range(this_patch.shape[1]):
                    for m in range(this_patch.shape[2]):
                        this_input = this_patch[i][j][m]
                        if layer_type == 'C':
                            wb = np.concatenate([this_input.reshape(1,-1), np.array([1.0]).reshape(1,-1)], axis = 1) # [1, 2305]
                        else:
                            wb = this_input.reshape(1, -1) # [1, 2304]
                        if use_tf_backend:
                            hessian_inverse = sess.run(Woodbury_hessian_inv_op, feed_dict={
                                hessian_inv_holder: hessian_inverse,
                                input_holder: wb
                            })
                        else:
                            denominator = dataset_size + np.dot(np.dot(wb,hessian_inverse), wb.T) 
                            numerator = np.dot(np.dot(hessian_inverse, wb.T), np.dot(wb,hessian_inverse))
                            hessian_inverse = hessian_inverse - numerator * (1.0 / denominator)
        
        if batch_idx % freq_moniter == 0:
            print '[%s] Now finish image No. %d / %d' \
                %(datetime.now(), batch_idx * batch_size, n_batch_used * batch_size)
    
        if batch_idx == n_batch_used:
            sess.close()
            break
    
    return hessian_inverse                