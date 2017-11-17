import tensorflow as tf
import numpy as np
import ctypes
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

num_ents = 38194
num_rels = 11
n = 100

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

ent_emb = sess.run('ent_emb:0')
rel_emb = sess.run('rel_emb:0')
Wh1 = sess.run('Wh1:0') 
Wh2 = sess.run('Wh2:0')
bias2 = sess.run('bias2:0')

validation_data = read_valid('./data/valid.txt')

with tf.device('/gpu:0'):
	
	h = tf.placeholder(tf.int32, [1])
	R = tf.placeholder(tf.int32, [1])
	l = tf.placeholder(tf.int32, [1])
	
	e1 = tf.squeeze(tf.nn.embedding_lookup(ent_emb, h))
	r  = tf.squeeze(tf.nn.embedding_lookup(rel_emb, R))
	e1_prime = tf.reshape(tf.tile(e1,[num_ents]),(num_ents,n))
	r_prime = tf.reshape(tf.tile(r,[num_ents]),(num_ents,n))
	batch = tf.concat([e1_prime,r_prime, ent_emb],axis=1)
	hl1 = tf.nn.relu(tf.matmul(batch,Wh1))
	op = tf.sigmoid(tf.matmul(hl1,Wh2) + bias2)[:,0]
	

	# e2 = tf.squeeze(tf.nn.embedding_lookup(ent_emb, l))
	# e2_prime = tf.reshape(tf.tile(e2,[num_ents]),(num_ents,n))
	# batch2 = tf.add(tf.concat([ent_emb, e2_prime],axis=1), r_prime)
	# hl1_ = tf.nn.relu(tf.matmul(batch2,Wh1))
	# op_ = tf.nn.sigmoid(tf.matmul(hl1_,Wh2) + bias2)[:,0]

	# with open('valid_tail_pred.txt','w') as fileout1, open('valid_head_pred.txt','w') as fileout2:
	# 	for t in validation_data:
	# 		pos1, pos2 = sess.run([op, op_] ,feed_dict={h:[t[0]], R:[t[1]], l:[t[2]]})
	# 		fileout1.write(' '.join([str(x) for x in pos1]) + '\n')
	# 		fileout2.write(' '.join([str(x) for x in pos2]) + '\n')

	with open('valid_tail_pred.txt','w') as fileout1:
		for t in validation_data:
			pos1 = sess.run(op ,feed_dict={h:[t[0]], R:[t[1]], l:[t[2]]})
			fileout1.write(' '.join([str(x) for x in pos1]) + '\n')
