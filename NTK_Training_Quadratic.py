# Code for training a two-layer Neural Tangent Kernel (NTK).
# In this example, the target function is a quadratic.
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
flags.DEFINE_boolean('add_bias', False, 'Whether bias is added to the input layer')

flags.DEFINE_enum('separation_regime', 'quadratic', ['quadratic', 'third_deg', 'linear'], 'The function generating the data')
flags.DEFINE_string('exp_name', 'test', 'The name of the experiment')

FLAGS = flags.FLAGS

# dimension of the input 
d = FLAGS.input_dim
gamma_mat = d / np.linspace(30, 450, num=15)
gamma = gamma_mat[FLAGS.gamma_ind]

# Number of hidden units
n = int(d / gamma)
alpha = FLAGS.alpha
beta = FLAGS.beta
directory = '/n_%d_%s.%s.%d_%.3f_square_d_%d'%(n, FLAGS.exp_name, FLAGS.separation_regime, FLAGS.job_id, alpha, d)
if not os.path.exists(directory):
	os.makedirs(directory)

np.random.seed(100)  
gamma = np.random.exponential(1, size=(d,))
meanGamma = np.array([np.sum(gamma)])
generator = None
if FLAGS.separation_regime == 'quadratic':
	def generator():
		while True:
			x = np.random.normal(size=(d,))    
			y = (np.sum(np.multiply(x ** 2, gamma))  - meanGamma) / np.sqrt(d)
			yield (x, y)
else:
	raise Exception('Function Not Implemented Yet')

lossval = np.zeros((FLAGS.num_iters,))
cevol = np.zeros((FLAGS.num_iters, 7))
grads = np.zeros((FLAGS.num_iters, 3))

def lr_fun(step):
	if step < (FLAGS.num_iters - 20000):
		return FLAGS.learning_rate
	if step < (FLAGS.num_iters - 10000):
		return FLAGS.learning_rate / 10.0	
	return FLAGS.learning_rate / 15.0

W0 = np.random.normal(size=(d, n)) / np.sqrt(d)
a0 = np.random.normal(size=(n, 1)) #/ np.sqrt(n)
U0 = W0 * a0.T * 2.0


g = tf.Graph()
tf.reset_default_graph()
with g.as_default():
	tf.set_random_seed(91)
	dataset = tf.data.Dataset.from_generator(generator, output_shapes=((d), (1)), output_types=(tf.float32, tf.float32)).repeat()                                       
	dataset = dataset.batch(FLAGS.batch_size).prefetch(1000)
	iter = dataset.make_one_shot_iterator()
	x, y = iter.get_next()
	
	w = tf.constant(W0, dtype=tf.float32, shape=[d, n], name='fixed_w')
	U = tf.constant(U0, dtype=tf.float32, shape=[d, n], name='fixed_aw')        

	at = tf.get_variable(initializer=tf.constant_initializer(a0), name='layer2', shape=[n, 1], dtype=tf.float32)	
	G = tf.get_variable(initializer=tf.constant_initializer(0.0), name='layer1_var', shape=[n, d], dtype=tf.float32)
	c = tf.get_variable(initializer=tf.constant_initializer(0.0), name='layer2_bias', shape=[], dtype=tf.float32)

	q = tf.matmul(x, w) ** 2 
	q2 = tf.reduce_sum(tf.matmul(x, tf.matmul(U, G)) * x, axis=1, keep_dims=True)
	yhat = alpha / (n + 0.0) * tf.matmul(q, at) + alpha / (n + 0.0) * q2 + beta * c

	loss_vec = (y - yhat) ** 2
	loss = tf.reduce_mean(loss_vec)
	lr = tf.placeholder(tf.float32, shape=[])
	if FLAGS.use_momentum:
		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)		
	else:
		opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
	
	grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
	grad_norm_w = tf.norm(tf.reshape(grads_and_vars[1][0], [-1])) 
	grad_norm_c = tf.norm(tf.reshape(grads_and_vars[2][0], [-1])) 
	grad_norm_a = tf.norm(tf.reshape(grads_and_vars[0][0], [-1])) 

	assert grads_and_vars[0][1].op.name =='layer2' # a_tilde
	assert grads_and_vars[1][1].op.name =='layer1_var' #G
	assert grads_and_vars[2][1].op.name =='layer2_bias' #c
	
	coeff = n * FLAGS.eta_a / (alpha ** 2 + 0.0)
	grads_and_vars[0] = (coeff * grads_and_vars[0][0], grads_and_vars[0][1])
	coeff = FLAGS.eta_w * n / (alpha ** 2 + 0.0)
	grads_and_vars[1] = (coeff * grads_and_vars[1][0], grads_and_vars[1][1])
	coeff = FLAGS.eta_c / (beta ** 2 + 0.0)
	grads_and_vars[2] = (coeff * grads_and_vars[2][0], grads_and_vars[2][1])
		
	train_op = opt.apply_gradients(grads_and_vars)	
	# Train the network
	with tf.Session() as sess:
	        sess.run(tf.global_variables_initializer())
	        G0, c0, a0 = sess.run([G, c, at])  
	        for i in range(FLAGS.num_iters):
			lossval[i], _, cevol[i, 0], atemp, grads[i, 0], grads[i, 1], grads[i, 2] = sess.run(\
				[loss, train_op, c, at, grad_norm_w, grad_norm_c, grad_norm_a], feed_dict={lr: lr_fun(i)})
			cevol[i, 1] = np.mean(atemp)
			if i % 200 == 0:		                
		                if i == 0:
		                	temp = 0
	                	else:
	                		i0 = np.maximum(0, i - 50)
	                		temp = np.mean(lossval[i0:i])
		                print('Iteration %d, Train loss %.3f, Smoothed loss %.3f'%(i, lossval[i], temp))
				f.flush()
				Gp, ap = sess.run([G, at])
				cevol[i, 2] = np.linalg.norm(Gp - G0)
				cevol[i, 3] = np.linalg.norm(ap - a0)
				cevol[i, 4] = np.mean(np.abs(ap))
				cevol[i, 5] = np.mean(ap)
				cevol[i, 6] = np.linalg.norm(Gp)
	        G1, c1, a1 = sess.run([G, c, at])      

save_dict = {}
for key in FLAGS.__flags.keys():
	save_dict[key] = getattr(FLAGS, key)

save_dict['loss'] = lossval
save_dict['c_evol'] = cevol
save_dict['gnorms'] = grads
save_dict['w0'] = G0
save_dict['a0'] = a0
save_dict['c0'] = c0

save_dict['w1'] = G1
save_dict['a1'] = a1
save_dict['c1'] = c1

fileName = directory + "/" + 'train_stats.pkl'
with open(fileName, 'wb') as output:    
	pickle.dump(save_dict, output, -1)