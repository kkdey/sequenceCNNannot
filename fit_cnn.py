import h5py
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import sklearn.metrics
import scipy.io
import collections
import os
from os import path
import argparse
import fnmatch


state = "even"
#feature="EnhA2"
#annot1_name = "E042-chrom_" + feature
#annot2_name = "E082-chrom_" + feature
feature = "TFBS_ENCODE"
#annot1_name = feature
#annot2_name = feature+ "_complement"
annot1_name = "TFBS_ENCODE"
annot2_name = "Repressed_Hoffman"
in_cell = "/n/groups/price/kushal/LDSC-Average/data/Genomewide_Sequences"
in_folder = in_cell + "/" + annot1_name + "_" + annot2_name
out_cell = "/n/groups/price/kushal/LDSC-Average/data/DeepTrain"
train_folder = out_cell + "/" + annot1_name + "_" + annot2_name + "_" + state


common_artificial_data = h5py.File(in_folder + "/train_" + state + ".h5", 'r')
trainxdata = common_artificial_data.get('trainxdata')
trainydata = common_artificial_data.get('trainydata')

######################. Convolutional Layer 1 configuration.  #############################
conv1_size = 4
num_kernels_1 = 5

#####################. Convolutional Layer 1 configuration.  ##############################
conv2_size = 4
num_kernels_2 = 5

#####################. Fully connected Layer 1 configuration. ##########################
fc_size_1 = 10


learning_rate = 0.01
dropout_keep_rate = 0.5

if not os.path.exists(train_folder):
	os.mkdir(train_folder)

tf.reset_default_graph()
x = tf.placeholder("float", [None, trainxdata.shape[1], trainxdata.shape[2]])
##. <tf.Tensor 'Placeholder:0' shape=(?, 1000, 4) dtype=float32>
y = tf.placeholder("float", [None, 2])
## <tf.Tensor 'Placeholder_1:0' shape=(?, 2) dtype=float32>
keep_prob = tf.placeholder(tf.float32)
## <tf.Tensor 'Placeholder_2:0' shape=<unknown> dtype=float32>
x_channels = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1, x.get_shape().as_list()[2]])

with tf.name_scope('conv1') as scope:
	W1 = tf.Variable(tf.truncated_normal([conv1_size, 1, trainxdata.shape[2], num_kernels_1], stddev=0.04))
	b1 = tf.Variable(tf.constant(0.0, shape=[num_kernels_1]))
	conv1 = tf.nn.conv2d(x_channels, W1, strides=[1, 1, 1, 1], padding='VALID')
	h_conv_1 = tf.nn.relu(conv1 + b1)
	h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

# with tf.name_scope('conv2') as scope:
# 	W2 = tf.Variable(tf.truncated_normal([conv2_size, 1, num_kernels_1, num_kernels_2], stddev=0.04))
# 	b2 = tf.Variable(tf.constant(0.0, shape=[num_kernels_2]))
# 	conv2 = tf.nn.conv2d(h_pool_1, W2, strides=[1, 1, 1, 1], padding='VALID')
# 	h_conv_2 = tf.nn.relu(conv2 + b2)
# 	h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

with tf.name_scope('fc1') as scope:
	h_pool_3_flat = tf.layers.flatten(h_pool_1)
	W_fc1 = tf.Variable(tf.truncated_normal([h_pool_3_flat.get_shape().as_list()[1], fc_size_1], stddev=0.04))
	b_fc1 = tf.Variable(tf.constant(0.0, shape=[fc_size_1]))
	h_fc1 = tf.nn.relu(tf.matmul(h_pool_3_flat, W_fc1) + b_fc1)
	dropout = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2') as scope:
	W_fc2 = tf.Variable(tf.truncated_normal([fc_size_1, 2], stddev=0.04))
	b_fc2 = tf.Variable(tf.constant(0.0, shape=[2]))
	h_fc2 = tf.matmul(dropout, W_fc2) + b_fc2
	ypred = tf.nn.sigmoid(h_fc2)

with tf.name_scope('train') as scope:
	loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	        logits=h_fc2, labels=y))
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

with tf.name_scope('accuracy') as scope:
	correct_pred = tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

num_start = 0
total_steps = 50000
save_model_steps=500
batch_size = 64
validxdata = trainxdata[-1000:,:,:]
validydata = trainydata[-1000:,:]

for i in range(num_start, num_start+total_steps):
	batch_start_index = np.asscalar(np.asarray(random.sample(range(0, trainxdata.shape[0] - 1000), 1)))
	batch_X = trainxdata[range(batch_start_index, batch_start_index + batch_size),:,:].astype(np.float32)
	batch_Y = trainydata[range(batch_start_index, batch_start_index + batch_size),:].astype(np.float32)
	sess.run(train_step, feed_dict={x: batch_X, y: batch_Y, keep_prob: dropout_keep_rate})
	loss, acc = sess.run([loss_op, accuracy],  feed_dict={x: batch_X, y: batch_Y, keep_prob: dropout_keep_rate})
	# print("Step " + str(i) + ", Training Loss= " + \
	#               "{:.4f}".format(loss) + ", Training Accuracy= " + \
	#               "{:.3f}".format(acc))
	if i%save_model_steps == 0 and i> 1:
		save_path = saver.save(sess, train_folder + "/model-tensorflow-" + str(i) + ".ckpt")
		print("Model saved at checkpoint at %d in path: %s" % (i, save_path))
		j=i-3
		preds_vec = sess.run(ypred,  feed_dict={x: validxdata, y: validydata, keep_prob: 1})
		fpr, tpr, thresholds = sklearn.metrics.roc_curve(validydata[:,0], preds_vec[:,0], pos_label=1)
		auc_val = sklearn.metrics.auc(fpr, tpr)
		if os.path.exists(train_folder + "/model-tensorflow-" + str(j) + ".ckpt"):
			os.remove(train_folder + "/model-tensorflow-" + str(j) + ".ckpt")
		print("Step " + str(i) + ", Validation AUC= " + \
	              "{:.3f}".format(auc_val))

