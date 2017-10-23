import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):
	'''Decorator for composable network layers.'''

	def layer_decorated(self, *args, **kwargs):
		# Automatically set a name if not provided.
		name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
		# Figure out the layer inputs.
		if len(self.terminals) == 0:
			raise RuntimeError('No input variables found for layer %s.' % name)
		elif len(self.terminals) == 1:
			layer_input = self.terminals[0]
		else:
			layer_input = list(self.terminals)
		# Perform the operation and get the output.
		layer_output = op(self, layer_input, *args, **kwargs)
		# Add to layer LUT.
		self.layers[name] = layer_output
		# This output is now the input for the next layer.
		self.feed(layer_output)
		# Return self for chained calls.
		return self

	return layer_decorated


class Network(object):

	def __init__(self, inputs, trainable=True):
		# The input nodes for this network
		self.inputs = inputs
		# The current list of terminal nodes
		self.terminals = []
		# Mapping from layer names to layers
		self.layers = dict(inputs)
		# If true, the resulting variables are set as trainable
		self.trainable = trainable
		# Switch variable for dropout
		self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
													   shape=[],
													   name='use_dropout')

		# Added op by Shangyu
		self.get_batch_hessian_op = dict()
		self.get_layer_inputs_op = dict()

		self.setup()

	def setup(self):
		'''Construct the network. '''
		raise NotImplementedError('Must be implemented by the subclass.')

	def load(self, data_path, session, ignore_missing=False):
		'''Load network weights.
		data_path: The path to the numpy-serialized network weights
		session: The current TensorFlow session
		ignore_missing: If true, serialized weights for missing layers are ignored.
		'''
		if type(data_path) == str:
			data_dict = np.load(data_path).item()
		elif type(data_path) == np.ndarray:
			data_dict = data_path
		for op_name in data_dict:
			with tf.variable_scope(op_name, reuse=True):
				if type(data_dict[op_name]) == dict:
					for param_name, data in data_dict[op_name].iteritems():
						try:
							var = tf.get_variable(param_name)
							session.run(var.assign(data))
						except ValueError:
							if not ignore_missing:
								raise
				else:
					var_weights = tf.get_variable('weights')
					session.run(var_weights.assign(data_dict[op_name][0]))

					var_biases = tf.get_variable('biases')
					session.run(var_biases.assign(data_dict[op_name][1]))

	def feed(self, *args):
		'''Set the input(s) for the next operation by replacing the terminal nodes.
		The arguments can be either layer names or the actual layers.
		'''
		assert len(args) != 0
		self.terminals = []
		for fed_layer in args:
			if isinstance(fed_layer, basestring):
				try:
					fed_layer = self.layers[fed_layer]
				except KeyError:
					raise KeyError('Unknown layer name fed: %s' % fed_layer)
			self.terminals.append(fed_layer)
		return self

	def get_output(self):
		'''Returns the current network output.'''
		return self.terminals[-1]

	def get_unique_name(self, prefix):
		'''Returns an index-suffixed unique name for the given prefix.
		This is used for auto-generating layer names based on the type-prefix.
		'''
		ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
		return '%s_%d' % (prefix, ident)

	def make_var(self, name, shape):
		'''Creates a new TensorFlow variable.'''
		return tf.get_variable(name, shape, trainable=self.trainable)

	def validate_padding(self, padding):
		'''Verifies that the padding is one of the supported ones.'''
		assert padding in ('SAME', 'VALID')

	@layer
	def conv(self,
			 input,
			 k_h,
			 k_w,
			 c_o,
			 s_h,
			 s_w,
			 name,
			 relu=True,
			 padding=DEFAULT_PADDING,
			 group=1,
			 biased=True):
		# Verify that the padding is acceptable
		self.validate_padding(padding)
		# Get the number of channels in the input
		c_i = input.get_shape()[-1]
		# Verify that the grouping parameter is valid
		assert c_i % group == 0
		assert c_o % group == 0
		# Convolution for a given input and kernel
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
		with tf.variable_scope(name) as scope:
			# Get input patches and construct hessian op
			# print '%s' %name
			get_patches_op = tf.extract_image_patches(input, \
													  ksizes=[1, k_h, k_w, 1], \
													  strides=[1, s_h, s_w, 1], \
													  rates=[1, 1, 1, 1], padding=padding)
			self.get_layer_inputs_op[name] = get_patches_op
			print 'Layer %s, input shape: %s' %(name, get_patches_op.get_shape())
			# print 'Input shape: %s' % input.get_shape().as_list()
			# First method to calculate hessain
			# print 'Patch shape: %s' % get_patches_op.get_shape().as_list()
			# patches_shape = get_patches_op.get_shape().as_list()
			# n_patches =  batch_size * patches_shape[1] * patches_shape[2] # Number of patches in one batch
			'''
			a = tf.expand_dims(get_patches_op, axis=-1)
			# print 'a shape: %s' %a.get_shape()
			a = tf.concat([a, tf.ones([tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2], 1, 1])], axis=3)
			# print 'a shape: %s' %a.get_shape()

			# print 'get_patches_op shape: %s' %get_patches_op.get_shape()
			b = tf.expand_dims(get_patches_op, axis=3)
			b = tf.concat([b, tf.ones([tf.shape(b)[0], tf.shape(b)[1], tf.shape(b)[2], 1, 1])], axis=4)
			# print 'b shape: %s' %b.get_shape()
			outprod = tf.multiply(a, b)
			# print 'outprod shape: %s' %outprod.get_shape()
			self.get_batch_hessian_op[name] = tf.reduce_mean(outprod, axis=[0, 1, 2])
			print 'Layer %s, hessian shape: %s' % (name, self.get_batch_hessian_op[name].get_shape())
			'''
			'''
			patches_shape = get_patches_op.get_shape().as_list()
			Dtensor = tf.reshape(get_patches_op, [-1, patches_shape[1] * patches_shape[2], patches_shape[3], 1])
			print 'Dtensor: %s' % Dtensor.get_shape()
			Dtensor = tf.concat([Dtensor, tf.ones([tf.shape(Dtensor)[0], tf.shape(Dtensor)[1], 1, 1])], axis=2)
			print 'Dtensor after concatenating one: %s' % Dtensor.get_shape()
			print 'Dtensor shape: %s' % Dtensor.get_shape()
			self.get_batch_hessian_op[name] = tf.reduce_mean(
				tf.matmul(Dtensor, Dtensor, transpose_b=True), axis=[0, 1])
			print 'Hessian shape: %s' % self.get_batch_hessian_op[name].get_shape()
			'''
			kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
			if group == 1:
				# This is the common-case. Convolve the input without any further complications.
				output = convolve(input, kernel)
			else:
				# Split the input into groups and then convolve each of them independently
				input_groups = tf.split(3, group, input)
				kernel_groups = tf.split(3, group, kernel)
				output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
				# Concatenate the groups
				output = tf.concat(3, output_groups)
			# Add the biases
			if biased:
				biases = self.make_var('biases', [c_o])
				output = tf.nn.bias_add(output, biases)
			if relu:
				# ReLU non-linearity
				output = tf.nn.relu(output, name=scope.name)
			return output

	@layer
	def relu(self, input, name):
		return tf.nn.relu(input, name=name)

	@layer
	def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.max_pool(input,
							  ksize=[1, k_h, k_w, 1],
							  strides=[1, s_h, s_w, 1],
							  padding=padding,
							  name=name)

	@layer
	def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.avg_pool(input,
							  ksize=[1, k_h, k_w, 1],
							  strides=[1, s_h, s_w, 1],
							  padding=padding,
							  name=name)

	@layer
	def lrn(self, input, radius, alpha, beta, name, bias=1.0):
		return tf.nn.local_response_normalization(input,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias,
												  name=name)

	@layer
	def concat(self, inputs, axis, name):
		return tf.concat(concat_dim=axis, values=inputs, name=name)

	@layer
	def add(self, inputs, name):
		return tf.add_n(inputs, name=name)

	@layer
	def fc(self, input, num_out, name, relu=True):
		with tf.variable_scope(name) as scope:
			input_shape = input.get_shape()
			if input_shape.ndims == 4:
				# The input is spatial. Vectorize it first.
				dim = 1
				for d in input_shape[1:].as_list():
					dim *= d
				feed_in = tf.reshape(input, [-1, dim])
			else:
				feed_in, dim = (input, input_shape[-1].value)
			self.get_layer_inputs_op[name] = feed_in
			print 'Layer %s, input shape: %s' % (name, feed_in.get_shape())
			weights = self.make_var('weights', shape=[dim, num_out])
			biases = self.make_var('biases', [num_out])
			op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
			fc = op(feed_in, weights, biases, name=scope.name)
			return fc

	@layer
	def softmax(self, input, name):
		input_shape = map(lambda v: v.value, input.get_shape())
		if len(input_shape) > 2:
			# For certain models (like NiN), the singleton spatial dimensions
			# need to be explicitly squeezed, since they're not broadcast-able
			# in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
			if input_shape[1] == 1 and input_shape[2] == 1:
				input = tf.squeeze(input, squeeze_dims=[1, 2])
			else:
				raise ValueError('Rank 2 tensor input expected for softmax!')
		return tf.nn.softmax(input, name=name)

	@layer
	def batch_normalization(self, input, name, scale_offset=True, relu=False):
		# NOTE: Currently, only inference is supported
		with tf.variable_scope(name) as scope:
			shape = [input.get_shape()[-1]]
			if scale_offset:
				scale = self.make_var('scale', shape=shape)
				offset = self.make_var('offset', shape=shape)
			else:
				scale, offset = (None, None)
			output = tf.nn.batch_normalization(
				input,
				mean=self.make_var('mean', shape=shape),
				variance=self.make_var('variance', shape=shape),
				offset=offset,
				scale=scale,
				# TODO: This is the default Caffe batch norm eps
				# Get the actual eps from parameters
				variance_epsilon=1e-5,
				name=name)
			if relu:
				output = tf.nn.relu(output)
			return output

	@layer
	def dropout(self, input, keep_prob, name):
		keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
		return tf.nn.dropout(input, keep, name=name)
