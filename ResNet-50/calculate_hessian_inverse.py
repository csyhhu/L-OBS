"""
This code calculate hessian inverse for various layers in ResNet-50

ResNet-50 contains 3 types of layer:
res layer: convolution layers without biases (e.g. res2a_branch2a)
convolution layer: normal convolution layers (e.g. conv1)
fully connected layer: fc1000

Somethings to be noticed:
1. If you find running too slow or memory explosive, please set batch_size in models/helpers a small number.
"""

import numpy as np
import tensorflow as tf

import models
import dataset
from datetime import datetime
import os

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


# Load the network
net = load_model('ResNet50')

# Load the dataset
ground_true = '../train_shuffle.txt' # Specify your imagenet groundtrue here
imagenet_data_dir = '/DATA4000A/imagenet/ILSVRC/Data/CLS-LOC/train' # Specify your imagenet dataset root here
model_path = 'res50.npy' # Specify your model parameters path here
data_spec = models.get_data_spec(model_instance=net)
image_producer = dataset.ImageNetProducer(val_path=ground_true,
										  data_path=imagenet_data_dir,
										  data_spec=data_spec)

layer_name_list = net.get_layer_inputs_op.keys()
# hessian = dict()
# layer_inputs = dict()
input_node = net.inputs['data']
# get_batch_hessian_op = net.get_batch_hessian_op
get_layer_inputs_op = net.get_layer_inputs_op
# In fact, L-OBS does not need all images to generate a good enough hessian, you can set n_batch
n_batch = 4000

def calculate_hessian_conv_tf(layer_inputs):
	"""
	This function calculates hessian for convolution layer, given a batch of layer inputs
	:param layer_inputs:
	:return: The hessin of this batch of data
	"""
	a = tf.expand_dims(layer_inputs, axis=-1)
	# print 'a shape: %s' %a.get_shape()
	a = tf.concat([a, tf.ones([tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2], 1, 1])], axis=3)
	# print 'a shape: %s' %a.get_shape()

	# print 'get_patches_op shape: %s' %get_patches_op.get_shape()
	b = tf.expand_dims(layer_inputs, axis=3)
	b = tf.concat([b, tf.ones([tf.shape(b)[0], tf.shape(b)[1], tf.shape(b)[2], 1, 1])], axis=4)
	# print 'b shape: %s' %b.get_shape()
	outprod = tf.multiply(a, b)
	# print 'outprod shape: %s' %outprod.get_shape()
	return tf.reduce_mean(outprod, axis=[0, 1, 2])


def calculate_hessian_res_tf(layer_inputs):
	"""
	This function calculates hessian for res layer, given a batch of layer inputs
	:param layer_inputs:
	:return:
	"""
	a = tf.expand_dims(layer_inputs, axis=-1)
	# print 'a shape: %s' %a.get_shape()

	# print 'get_patches_op shape: %s' %get_patches_op.get_shape()
	b = tf.expand_dims(layer_inputs, axis=3)

	outprod = tf.multiply(a, b)
	# print 'outprod shape: %s' %outprod.get_shape()
	return tf.reduce_mean(outprod, axis=[0, 1, 2])


def calculate_hessian_fc_tf(layer_inputs):
	"""
	This function calculates hessian for fully-connected layer, given a batch of layer inputs
	:param layer_inputs:
	:return:
	"""
	a = tf.expand_dims(layer_inputs, axis=-1)
	# print 'a shape: %s' %a.get_shape()
	a = tf.concat([a, tf.ones([tf.shape(a)[0], 1, 1])], axis=1)
	# print 'a shape: %s' %a.get_shape()

	# print 'get_patches_op shape: %s' %get_patches_op.get_shape()
	b = tf.expand_dims(layer_inputs, axis=1)
	b = tf.concat([b, tf.ones([tf.shape(b)[0], 1, 1])], axis=2)
	# print 'b shape: %s' %b.get_shape()
	outprod = tf.multiply(a, b)
	# print 'outprod shape: %s' %outprod.get_shape()
	return tf.reduce_mean(outprod, axis=0)

# Define hessian calculation operators
layer_inputs_holder = tf.placeholder(dtype=tf.float32)
get_hessian_conv_op = calculate_hessian_conv_tf(layer_inputs_holder)
get_hessian_res_op = calculate_hessian_res_tf(layer_inputs_holder)
get_hessian_fc_op = calculate_hessian_fc_tf(layer_inputs_holder)

for layer_name in layer_name_list:
	print '[%s] Now process layer %s' % (datetime.now(), layer_name)
	with tf.Session() as sesh:
		# Load the converted parameters
		net.load(data_path=model_path, session=sesh)
		coordinator = tf.train.Coordinator()
		# Start the image processing workers
		threads = image_producer.start(session=sesh, coordinator=coordinator)
		# Iterate over and classify mini-batches
		for batch_count, (labels, images) in enumerate(image_producer.batches(sesh)):
			# Get layer inputs for this layer
			layer_inputs = sesh.run(get_layer_inputs_op[layer_name], feed_dict={input_node: images})
			# print layer_inputs.shape
			if layer_name.startswith('conv'):
				if batch_count == 0:
					hessian = sesh.run(get_hessian_conv_op, feed_dict={layer_inputs_holder: layer_inputs})
				else:
					hessian += sesh.run(get_hessian_conv_op, feed_dict={layer_inputs_holder: layer_inputs})
			elif layer_name.startswith('res'):
				if batch_count == 0:
					hessian = sesh.run(get_hessian_res_op, feed_dict={layer_inputs_holder: layer_inputs})
				else:
					hessian += sesh.run(get_hessian_res_op, feed_dict={layer_inputs_holder: layer_inputs})
			elif layer_name.startswith('fc'):
				if batch_count == 0:
					hessian = sesh.run(get_hessian_fc_op, feed_dict={layer_inputs_holder: layer_inputs})
				else:
					hessian += sesh.run(get_hessian_fc_op, feed_dict={layer_inputs_holder: layer_inputs})
			# See watch info
			print '[%s] Now process to batch %d.' %(datetime.now(), batch_count)

			# In fact, L-OBS does not need all images to generate a good enough hessian, you can set n_batch
			if batch_count == n_batch:
				break

		print '[%s] Hessian calculated finish Now calculate its inverse.' % datetime.now()
		hessian_inverse = np.linalg.pinv(hessian)
		print '[%s] Hessian inverse calculated finish. Now saving.' % datetime.now()
		if not os.path.exists('hessian_inverse'):
			os.makedirs('hessian_inverse')
		np.save('hessian_inverse/%s' %layer_name, hessian_inverse)
		# Stop the worker threads
		coordinator.request_stop()
		coordinator.join(threads, stop_grace_period_secs=2)