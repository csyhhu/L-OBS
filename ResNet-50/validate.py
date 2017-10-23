#!/usr/bin/env python
'''Validates a converted ImageNet model against the ILSVRC12 validation set.'''

import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset

tf.reset_default_graph()

def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': data_node})


def validate(net, model_path, image_producer, top_k=5):
    '''Compute the top_k classification accuracy for the given network and images.'''
    # Get the data specifications for given network
    spec = models.get_data_spec(model_instance=net)
    # Get the input node for feeding in the images
    input_node = net.inputs['data']
    # Create a placeholder for the ground truth labels
    label_node = tf.placeholder(tf.int32)
    # Get the output of the network (class probabilities)
    probs = net.get_output()
    # Create a top_k accuracy node
    top_k_op = tf.nn.in_top_k(probs, label_node, top_k)
    # The number of images processed
    count = 0
    # The number of correctly classified images
    correct = 0
    # The total number of images
    total = len(image_producer)

    with tf.Session() as sesh:
        coordinator = tf.train.Coordinator()
        # Load the converted parameters
        net.load(data_path=model_path, session=sesh)
        # Start the image processing workers
        threads = image_producer.start(session=sesh, coordinator=coordinator)
        # Iterate over and classify mini-batches
        for (labels, images) in image_producer.batches(sesh):
            # print images.shape
            correct += np.sum(sesh.run(top_k_op,
                                       feed_dict={input_node: images,
                                                  label_node: labels}))
            count += len(labels)
            cur_accuracy = float(correct) * 100 / count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
    print('Top {} Accuracy: {}'.format(top_k, float(correct) / total))



def make_pruned_parameters(parameters, pruned_weights_root):

    for layer_name in parameters.keys():

        if layer_name.startswith('res'):
            parameters[layer_name]['weights'] = np.load('%s/%s/weights.npy' %(pruned_weights_root, layer_name))
        elif layer_name.startswith('conv') or layer_name.startswith('fc'):
            parameters[layer_name]['weights'] = np.load('%s/%s/weights.npy' % (pruned_weights_root, layer_name))
            parameters[layer_name]['biases'] = np.load('%s/%s/biases.npy' % (pruned_weights_root, layer_name))

    return parameters


# Load the network
net = load_model('ResNet50')
if net is None:
    exit(-1)
exit(-1)

parameters = np.load('res50.npy')
# Place the pruned weights into a parameters
pruned_parameters = make_pruned_parameters(parameters, 'pruned_weights')
# Load the dataset
val_gt = '../val.txt'
# imagenet_data_dir = '../val_img/'
imagenet_data_dir = '/DATA4000A/imagenet/ILSVRC/Data/CLS-LOC/val'
# model_path = 'vgg16.npy'
data_spec = models.get_data_spec(model_instance=net)
image_producer = dataset.ImageNetProducer(val_path=val_gt,
                                          data_path=imagenet_data_dir,
                                          data_spec=data_spec)

# Evaluate its performance on the ILSVRC12 validation set
validate(net, pruned_parameters, image_producer)

