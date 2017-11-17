import tensorflow as tf
import numpy as np
import ctypes
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def read_valid(filename):
	valid_triples = []
	with open(filename,'r') as filein:
		for line in filein:
			valid_triples.append([int(x.strip()) for x in line.split()])
	return valid_triples

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state(os.path.dirname('logs/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
	saver.restore(sess, ckpt.model_checkpoint_path)

ent_real = sess.run('ent_real:0')
ent_img = sess.run('ent_img:0')
rel_real = sess.run('rel_real:0')
rel_img = sess.run('rel_img:0')

validation_data = read_valid('./data/valid.txt')

with tf.device('/gpu:1'):
	
	h = tf.placeholder(tf.int32, [1])
	R = tf.placeholder(tf.int32, [1])
	l = tf.placeholder(tf.int32, [1])
	
	e1_real = tf.squeeze(tf.nn.embedding_lookup(ent_real, h))
	e1_img = tf.squeeze(tf.nn.embedding_lookup(ent_img, h))
	r_real  = tf.squeeze(tf.nn.embedding_lookup(rel_real, R))
	r_img  = tf.squeeze(tf.nn.embedding_lookup(rel_img, R))
	e2_real = tf.squeeze(tf.nn.embedding_lookup(ent_real, l))
	e2_img = tf.squeeze(tf.nn.embedding_lookup(ent_img, l))

	score1 = tf.reduce_sum(tf.multiply(tf.multiply(r_real, e1_real),ent_real), axis=1)
	score2 = tf.reduce_sum(tf.multiply(tf.multiply(r_real, e1_img),ent_img), axis=1)
	score3 = tf.reduce_sum(tf.multiply(tf.multiply(r_img, e1_real),ent_img), axis=1)
	score4 = tf.reduce_sum(tf.multiply(tf.multiply(r_img, e1_img),ent_real), axis=1)
	op = tf.nn.sigmoid(score1 + score2 + score3 - score4)
	
	# score1_ = tf.reduce_sum(tf.multiply(tf.multiply(r_real, ent_real),e2_real), axis=1)
	# score2_ = tf.reduce_sum(tf.multiply(tf.multiply(r_real, ent_img),e2_img), axis=1)
	# score3_ = tf.reduce_sum(tf.multiply(tf.multiply(r_img, ent_real),e2_img), axis=1)
	# score4_ = tf.reduce_sum(tf.multiply(tf.multiply(r_img, ent_img),e2_real), axis=1)
	# op_ = tf.nn.sigmoid(score1_ + score2_ + score3_ - score4_)

	# with open('valid_tail_pred.txt','w') as fileout1, open('valid_head_pred.txt','w') as fileout2:
	# 	for t in validation_data:
	# 		pos1, pos2 = sess.run([op, op_] ,feed_dict={h:[t[0]], R:[t[1]], l:[t[2]]})
	# 		fileout1.write(' '.join([str(x) for x in pos1]) + '\n')
	# 		fileout2.write(' '.join([str(x) for x in pos2]) + '\n')

	with open('valid_tail_pred.txt','w') as fileout1:
		for t in validation_data:
			pos1 = sess.run(op ,feed_dict={h:[t[0]], R:[t[1]], l:[t[2]]})
			fileout1.write(' '.join([str(x) for x in pos1]) + '\n')