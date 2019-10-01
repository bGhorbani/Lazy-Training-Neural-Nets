# Code for training a two-layer neural network with mean-field 
# parameterization. In this example, the target function is a quadratic.
# Last edited by Behrooz Ghorbani on 2019-10-01.
# ghorbani@stanford.edu

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os 
import math
import cPickle as pickle

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for SGD')

flags.DEFINE_float('alpha', 1.0, 'Scaling parameter alpha')
flags.DEFINE_float('beta', 1.0, 'Scaling parameter beta')

flags.DEFINE_float('eta_w', 1.0, 'Scaling parameter for step-size of w')
flags.DEFINE_float('eta_c', 1.0, 'Scaling parameter for step-size of c')
flags.DEFINE_float('eta_a', 1.0, 'Scaling parameter for step-size of a')

flags.DEFINE_integer('input_dim', 450, 'The input dimension, d')
flags.DEFINE_integer('gamma_ind', -1, 'Index for the value of Gamma')
flags.DEFINE_integer('batch_size', 100, 'Batch Size')
flags.DEFINE_integer('num_iters', 100000, 'Number of training steps')
flags.DEFINE_integer('job_id', -1, 'Unique job id assigned by the cluster')

flags.DEFINE_boolean('use_momentum', False, 'Whether momentum is used for training')
flags.DEFINE_boolean('train_w', True, 'Whether W is trained')
flags.DEFINE_boolean('add_bias', False, 'Whether bias is added to the input layer')
flags.DEFINE_boolean('constant_a', True, 'Whether the second layer is a multiple of ones')

flags.DEFINE_enum('separation_regime', 'quadratic', ['quadratic', 'third_deg', 'linear'], 'The function generating the data')
flags.DEFINE_enum('transform', 'square', ['square', 'relu', 'tanh'], 'The nonlinear transform used for the network')
flags.DEFINE_string('exp_name', 'test', 'The name of the experiment')

FLAGS = flags.FLAGS
d = FLAGS.input_dim
gamma_mat = d / (30 * 1.5 ** np.linspace(0, 12.5, num=25))
gamma = gamma_mat[FLAGS.gamma_ind]

# Number of hidden units
n = int(d / gamma)
alpha = FLAGS.alpha
beta = FLAGS.beta
directory = '/n_%d_d_%d_%s_%.4f_%s.%d'%(n, d, FLAGS.exp_name, FLAGS.learning_rate, FLAGS.separation_regime, FLAGS.job_id)
if not os.path.exists(directory):
	os.makedirs(directory)
np.random.seed(100)  
gamma = np.random.exponential(1, size=(1, d))
meanGamma = np.array([np.sum(gamma)])
generator = None
if FLAGS.separation_regime == 'quadratic':
	def generator():	    		
		x = np.random.normal(size=(30000, d))    
		y = (np.sum(np.multiply(np.power(x, 2), gamma), axis=1, keepdims=True)  - meanGamma) / np.sqrt(d)
		return (x, y)
else:
	raise Exception('Function Not Implemented Yet')

# Write the function generating the data
lossval = np.zeros((FLAGS.num_iters,))
cevol = np.zeros((FLAGS.num_iters, 7))
grads = np.zeros((FLAGS.num_iters, 3))

def lr_fun(step):
	if step < (FLAGS.num_iters - 30000):
		return FLAGS.learning_rate
	if step < (FLAGS.num_iters - 15000):
		return FLAGS.learning_rate / 10.0	
	return FLAGS.learning_rate / 15.0

g = tf.Graph()
tf.reset_default_graph()
with g.as_default():
	tf.set_random_seed(91)
	features_placeholder = tf.placeholder(tf.float32, (30000, d))
	labels_placeholder = tf.placeholder(tf.float32, (30000, 1))
	dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)).repeat(1).batch(FLAGS.batch_size)
	iterator = dataset.make_initializable_iterator()
	x, y = iterator.get_next()
	
	w = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(d)),\
	 name="layer1", shape=[d, n], dtype=tf.float32, trainable=FLAGS.train_w)
	if FLAGS.constant_a:
		a = tf.get_variable(initializer=tf.constant_initializer(0.0), name='layer2', shape=[], dtype=tf.float32)
	else:
		a = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0), name='layer2', shape=[n, 1], dtype=tf.float32)
	c = tf.get_variable(initializer=tf.constant_initializer(0.0), name='layer2_bias', shape=[], dtype=tf.float32)
	if FLAGS.add_bias:
		b = tf.get_variable(initializer=tf.constant_initializer(0.0), name='layer1_bias', shape=[1, n], dtype=tf.float32)
		z = tf.matmul(x, w) + b
	else:
		z = tf.matmul(x, w)
	
	if FLAGS.transform == 'square':
		q = z ** 2
	elif FLAGS.transform == 'relu':
		q = tf.nn.relu(z)
	elif FLAGS.transform == 'tanh':
		q = tf.nn.tanh(z)
	else:
		raise Exception(FLAGS.transform + ' is not valid')
	if FLAGS.constant_a:
		yhat = alpha * a / (n + 0.0) * tf.reduce_sum(q, axis=1, keepdims=True) + beta * c 		
	else:
		yhat = alpha / (n + 0.0) * tf.matmul(q, a) + beta * c

	loss_vec = (y - yhat) ** 2	
	loss = tf.reduce_mean(loss_vec)    
	lr = tf.placeholder(tf.float64, shape=[])
	opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
	
	grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
	grad_norm_w = tf.norm(tf.reshape(grads_and_vars[0][0], [-1])) 
	grad_norm_c = tf.norm(tf.reshape(grads_and_vars[2][0], [-1])) 
	grad_norm_a = tf.norm(tf.reshape(grads_and_vars[1][0], [-1])) 

	assert grads_and_vars[0][1].op.name =='layer1'
	assert grads_and_vars[1][1].op.name =='layer2'
	assert grads_and_vars[2][1].op.name =='layer2_bias'
	if FLAGS.constant_a:
		coeff = n * FLAGS.eta_w / (alpha ** 2 + 0.0)		
		grads_and_vars[0] = (coeff * grads_and_vars[0][0], grads_and_vars[0][1]) #W
		coeff = FLAGS.eta_a / (alpha ** 2 + 0.0)
		grads_and_vars[1] = (coeff * grads_and_vars[1][0], grads_and_vars[1][1]) #a
		coeff = FLAGS.eta_c / (beta ** 2 + 0.0)
		grads_and_vars[2] = (coeff * grads_and_vars[2][0], grads_and_vars[2][1]) #c
	else:
		coeff = n * FLAGS.eta_w / (alpha ** 2 + 0.0)
		grads_and_vars[0] = (coeff * grads_and_vars[0][0], grads_and_vars[0][1])
		coeff = FLAGS.eta_a * n / (alpha ** 2 + 0.0)
		grads_and_vars[1] = (coeff * grads_and_vars[1][0], grads_and_vars[1][1])
		coeff = FLAGS.eta_c / (beta ** 2 + 0.0)
		grads_and_vars[2] = (coeff * grads_and_vars[2][0], grads_and_vars[2][1])
	train_op = opt.apply_gradients(grads_and_vars)	
	# Train the network
	with tf.Session() as sess:
	        sess.run(tf.global_variables_initializer())
	        w0, c0, a0 = sess.run([w, c, a])
	        # initialize the iterator
	        newX, newY = generator()
		sess_dict = {features_placeholder: newX, labels_placeholder: newY}
                sess.run(iterator.initializer, feed_dict=sess_dict)
	        for i in range(FLAGS.num_iters):
	        	try:
				lossval[i], _, cevol[i, 0], atemp, grads[i, 0], grads[i, 1], grads[i, 2] = sess.run(\
					[loss, train_op, c, a, grad_norm_w, grad_norm_c, grad_norm_a], feed_dict={lr: lr_fun(i)})
				cevol[i, 1] = np.mean(atemp)
				if i % 200 == 0:		                
			                if i == 0:
			                	temp = 0
		                	else:
		                		i0 = np.maximum(0, i - 50)
		                		temp = np.mean(lossval[i0:i])
					
			                print('Iteration %d, Train loss %.3f, Smoothed loss %.3f'%(i, lossval[i], temp))					
					wp, ap = sess.run([w, a])
					cevol[i, 2] = np.linalg.norm(wp - w0)
					cevol[i, 3] = np.linalg.norm(ap - a0)
					cevol[i, 4] = np.mean(np.abs(ap))
					cevol[i, 5] = np.mean(ap)
					cevol[i, 6] = np.linalg.norm(wp)
			except tf.errors.OutOfRangeError:
				newX, newY = generator()
				sess_dict = {features_placeholder: newX, labels_placeholder: newY}
		                sess.run(iterator.initializer, feed_dict=sess_dict)
	        w1, c1, a1 = sess.run([w, c, a])

save_dict = {}
for key in FLAGS.__flags.keys():
	save_dict[key] = getattr(FLAGS, key)

save_dict['loss'] = lossval
save_dict['c_evol'] = cevol
save_dict['gnorms'] = grads
save_dict['w0'] = w0
save_dict['a0'] = a0
save_dict['c0'] = c0

save_dict['w1'] = w1
save_dict['a1'] = a1
save_dict['c1'] = c1

fileName = directory + "/" + 'train_stats.pkl'
with open(fileName, 'wb') as output:    
	pickle.dump(save_dict, output, -1)
