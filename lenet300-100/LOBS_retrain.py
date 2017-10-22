'''
This code combine hessian calculation and edge cut and retrain together in lenet300-100 with MNIST dataset.

The L-OBS works as following:
1) Generate

'''
import os
import numpy as np
import time
import tensorflow as tf

# Parameters
n_hidden_1 = 300 # 1st layer number of features
n_hidden_2 = 100 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1
display_batch_step = 10

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# mnist = np.load('mnist.npy')

def produce_output(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['fc2']), biases['fc2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	# out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return x, layer_1, layer_2

# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['fc2']), biases['fc2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['fc3']) + biases['fc3']
	return out_layer

def hessian_prune4fc(hessian_inverse_path ="", w_layer_path ="", layer_input_train_dir =""):

	"""
	:param hessian_inverse_path: the hessian inverse path you store
	:param w_layer_path: the layer weights
	:param layer_input_train_dir: layer inputs
	:return:
	"""

	w_layer = np.load(w_layer_path)
	# all_hessian_inverse = []
	# dataset_size = layer2_input_train.shape[0]
	n_hidden_1 = w_layer.shape[0]
	# n_hidden_2 = w_layer.shape[1]

	hessian_inverse = 1000000 * np.eye(n_hidden_1 + 1)

	jth = 0
	for j in os.listdir(layer_input_train_dir):   
		#start = time.time() 
		layer2_input_train = np.load(layer_input_train_dir + '/' + j)
		if jth == 0:
			dataset_size = layer2_input_train.shape[0]*len(os.listdir(layer_input_train_dir))
			# print 'dataset_size: %s.' % dataset_size
		jth += 1
		for i in range(layer2_input_train.shape[0]):    
			vect_w_b = np.vstack((np.array([layer2_input_train[i]]).T,np.array([[1.0]])))
			denominator = dataset_size + np.dot(np.dot(vect_w_b.T,hessian_inverse),vect_w_b)
			numerator = np.dot(np.dot(hessian_inverse,vect_w_b),np.dot(vect_w_b.T,hessian_inverse))
			hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)
		#all_hessian_inverse.append(hessian_inverse)
		#end = time.time()
		if jth % 100 == 0:
			#print 'Finish processing batch %s with time %s' %(jth, (end-start))
			print 'Finish processing batch %s' %(jth)
	np.save(hessian_inverse_path, hessian_inverse)
	print('Hessian Inverse Done!')

def edge_cut(sorted_index_path, gate_w_path, gate_b_path, hein_name, w_layer_path, b_layer_path, save_path_w, save_path_b, save_gate_w_path, save_gate_b_path, cut_ratio):
	
	# Argument:
	# sorted_index_path: the sorted index for all weights, to fasten process in case of interruption.
	# gate_w_path: gate of weights in the last iteration
	# gate_b_path: gate of biases in the last iteration
	# hein_name: hessian matrix path
	# w_layer: weights in last iteration
	# b_layer: biases in last iteration
	# save_path_w: path to save this iteration's weights
	# save_path_b: path to save this iteration's biases
	# cut_ratio: the ratio of weights cut this time, compared with the total weights

	# dataset_size = layer2_input_train.shape[0]
	w_layer = np.load(w_layer_path)
	b_layer = np.load(b_layer_path)
	n_hidden_1 = w_layer.shape[0]
	n_hidden_2 = w_layer.shape[1]
	
	L = np.array([])

	hessian_inverse = np.load(hein_name)
	print('Hessian Inverse Done!')
	# Load in gate w
	if os.path.isfile(gate_w_path) == False:
		print 'Can not find gate matrix for w, create one.'
		gate_w = np.ones([n_hidden_1,n_hidden_2]) 
	else:
		gate_w = np.load(gate_w_path)
	# Load in gate b 
	if os.path.isfile(gate_w_path) == False:
		print 'Can not find gate matrix for b, create one.'
		gate_b = np.ones([n_hidden_2])
	else:
		gate_b = np.load(gate_b_path)

	# dx pruned method
	max_pruned_num = int(n_hidden_1 * n_hidden_2 * cut_ratio)
	print 'Max prune number : %s' %max_pruned_num
	# sorted_index = heapq.nsmallest(max_pruned_num, range(len(L)), L.take)
	if os.path.isfile(sorted_index_path) == False:
		for i in range(n_hidden_2):
			L = np.hstack((L,0.5*(np.hstack((w_layer.T[i],b_layer[i]))**2)/np.diag(hessian_inverse)))
		sorted_index = np.argsort(L)
		print 'Sorted index generate completed.'
		np.save('sorted_index_for_hessian_weights_fc', sorted_index)   
	else:
		sorted_index = np.load(sorted_index_path)
		# print 'Sorted index read completed.'
	# dx_accuracy_pruned_list=[]                        #acc recorder
	print('Starting Pruning!')
	hessian_inverseT = hessian_inverse.T

	#current_ratio = 0

	#t = time.time()
	
	prune_count = 0
	for i in range(n_hidden_1 * n_hidden_2):
		'''
		# Every 5% cut increase, save the result
		cut_ratio = (i * 1.0) * 100 / (n_hidden_1 * n_hidden_2 * 1.0)

		if int(cut_ratio)  == current_ratio :
			print 'Now Process to %s ratio %s with time comsuming : %s' %(i, current_ratio, time.time() - t)
			current_ratio += 1
			t = time.time()
		'''
		prune_index = [sorted_index[i]]
		x_index = prune_index[0]/(n_hidden_1+1)   # next layer num
		y_index = prune_index[0]%(n_hidden_1+1)   # this layer num

		if y_index == n_hidden_1:  # b
			if gate_b[x_index] == 1:
				delta_w = (-b_layer[x_index]/hessian_inverse[y_index][y_index])*hessian_inverseT[y_index]	
				gate_b[x_index] = 0
				prune_count +=1;
				w_layer.T[x_index] = w_layer.T[x_index] + delta_w[0:-1]
				b_layer[x_index] = b_layer[x_index] + delta_w[-1]
		else:
			if gate_w[y_index][x_index] == 1:	
				delta_w = (-w_layer[y_index][x_index]/hessian_inverse[y_index][y_index])*hessian_inverseT[y_index]		
				gate_w[y_index][x_index] = 0
				prune_count +=1;
				w_layer.T[x_index] = w_layer.T[x_index] + delta_w[0:-1]
				b_layer[x_index] = b_layer[x_index] + delta_w[-1]

		w_layer = w_layer * gate_w
		b_layer = b_layer * gate_b

		if prune_count == max_pruned_num:
			print 'Have prune required weights'
			break;
	print 'Prune Finish. Zero percentage:'
	print 1 - float(np.count_nonzero(gate_w)/float(gate_w.shape[0] * gate_w.shape[1]))
	np.save(save_path_w, w_layer)
	np.save(save_path_b, b_layer)
	np.save(save_gate_w_path, gate_w)
	np.save(save_gate_b_path, gate_b)

# Retrain Iteration Loop
# Define the cut ratio in retrain iteration
# cut_ratio = [0.5, 0.4, 0.08]
cut_ratio = {
	'fc1': [0.93, 0.06, 0, 0],
	'fc2': [0.8, 0.16, 0, 0],
	'fc3': [0.3, 0.16, 0, 0]
}
for ite in range(1, 3):
	print 'Now begin iteration ' + str(ite)
	# If it is the first iteration, get the parameters from store: 100.spydata
	# Calculate hessian inverse
	if ite != 1:
		hessian_prune4fc(hessian_inverse_path='../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc1', \
						 w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer1.npy', \
						 layer_input_train_dir = '../fc1_input/iteration' + str(ite))
		print 'Complete fc1 hessian calculation.'
		hessian_prune4fc(hessian_inverse_path='../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc2', \
						 w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer2.npy', \
						 layer_input_train_dir = '../fc2_input/iteration' + str(ite))
		print 'Complete fc2 hessian calculation.'
		hessian_prune4fc(hessian_inverse_path='../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc3', \
						 w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer3.npy', \
						 layer_input_train_dir = '../fc3_input/iteration' + str(ite))
		print 'Complete fc3 hessian calculation.'
	# Cut the edge
	edge_cut(sorted_index_path = '', gate_w_path = '../gate/iteration' + str(ite-1) + '/fc1_w_gate.npy', \
		gate_b_path = '../gate/iteration' + str(ite-1) + '/fc1_b_gate.npy', \
		hein_name = '../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc1.npy', \
		w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer1.npy', \
		b_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/b_layer1.npy', \
		save_path_w = '../new_weights/after_iteration' + str(ite) + '/w_layer1.npy', \
		save_path_b = '../new_weights/after_iteration' + str(ite) + '/b_layer1.npy', \
		save_gate_w_path = '../gate/iteration' + str(ite) + '/fc1_w_gate', \
		save_gate_b_path = '../gate/iteration' + str(ite) + '/fc1_b_gate', \
		cut_ratio = cut_ratio['fc1'][ite-1])
	print 'Complete fc1 edge cut.'
	edge_cut(sorted_index_path = '', gate_w_path = '../gate/iteration' + str(ite-1) + '/fc2_w_gate.npy', \
		gate_b_path = '../gate/iteration' + str(ite-1) + '/fc2_b_gate.npy', \
		hein_name = '../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc2.npy', \
		w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer2.npy', \
		b_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/b_layer2.npy', \
		save_path_w = '../new_weights/after_iteration' + str(ite) + '/w_layer2.npy', \
		save_path_b = '../new_weights/after_iteration' + str(ite) + '/b_layer2.npy', \
		save_gate_w_path = '../gate/iteration' + str(ite) + '/fc2_w_gate', \
		save_gate_b_path = '../gate/iteration' + str(ite) + '/fc2_b_gate', \
		cut_ratio = cut_ratio['fc2'][ite-1])
	print 'Complete fc2 edge cut.'
	edge_cut(sorted_index_path = '', gate_w_path = '../gate/iteration' + str(ite-1) + '/fc3_w_gate.npy', \
		gate_b_path = '../gate/iteration' + str(ite-1) + '/fc3_b_gate.npy', \
		hein_name = '../hessian_inverse/iteration' + str(ite) + '/hessian_inverse_fc3.npy', \
		w_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/w_layer3.npy', \
		b_layer_path = '../new_weights/after_iteration' + str(ite-1) + '/b_layer3.npy', \
		save_path_w = '../new_weights/after_iteration' + str(ite) + '/w_layer3.npy', \
		save_path_b = '../new_weights/after_iteration' + str(ite) + '/b_layer3.npy', \
		save_gate_w_path = '../gate/iteration' + str(ite) + '/fc3_w_gate', \
		save_gate_b_path = '../gate/iteration' + str(ite) + '/fc3_b_gate', \
		cut_ratio = cut_ratio['fc3'][ite-1])
	print 'Complete fc 3 edge cut.'

	# Test validation accuracy and generate new output
	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])
	# load layers weight & bias
	print 'Getting weights from ../new_weights/after_iteration' + str(ite)
	weights = {
		'fc1': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/w_layer1.npy').astype('float32')),
		'fc2': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/w_layer2.npy').astype('float32')),
		'fc3': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/w_layer3.npy').astype('float32'))
	}

	biases = {
		'fc1': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/b_layer1.npy').astype('float32')),
		'fc2': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/b_layer2.npy').astype('float32')),
		'fc3': tf.Variable(np.load('../new_weights/after_iteration' + str(ite) + '/b_layer3.npy').astype('float32'))
	}
	# Construct model
	pred = multilayer_perceptron(x, weights, biases)
	layer_outputs = produce_output(x, weights, biases)
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		total_batch = int(mnist.train.num_examples/batch_size)
		# Training cycle
		# Test the raw accuracy
		acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print("First time after pruning validation accuracy=", \
			"{:.9f}".format(acc))
		for epoch in range(training_epochs):
			avg_cost = 0.
			# Loop over all batches
			for i in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				# batch_x, batch_y = mnist[0].next_batch(batch_size)
				# Run optimization op (backprop) and cost op (to get loss value)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				# Compute average loss
				# avg_cost += c / total_batch
			# Display logs per epoch step
			#if epoch % display_step == 0:
				if i % display_batch_step == 0:
					acc,lost = sess.run([accuracy, cost], feed_dict={x: mnist.test.images, y: mnist.test.labels})
					print("Epoch:", '%04d' % (epoch+1), "lost=", "{:.9f}".format(lost), "Validation accuracy=", "{:.9f}".format(acc))
					if acc > 0.982:
						print('Reach acc at 0.982, break')
						break
		print("Optimization Finished!")

		# acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		# print("Validation accuracy=", "{:.9f}".format(acc))

		# Save this iteration's layer output
		# Loop over all batches
		print 'Now saving iteration' + str(ite) + ' s output:'
		for batch_index in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			fc1_input, fc2_input, fc3_input = sess.run(layer_outputs, feed_dict={x: batch_x, y: batch_y})
			np.save('../fc1_input/iteration' + str(ite) + '/batch-' + str(batch_index), fc1_input)
			np.save('../fc2_input/iteration' + str(ite) + '/batch-' + str(batch_index), fc2_input)
			np.save('../fc3_input/iteration' + str(ite) + '/batch-' + str(batch_index), fc3_input)
		print 'Saving output completed.'