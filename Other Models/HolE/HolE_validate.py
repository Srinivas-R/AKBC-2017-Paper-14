import tensorflow as tf
import numpy as np
import ctypes
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

num_ents = 14951
num_rels = 1345
n = 200

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

validation_data = read_valid('./data/valid.txt')

with tf.device('/gpu:2'):
	
	h = tf.placeholder(tf.int32, [1])
	R = tf.placeholder(tf.int32, [1])
	l = tf.placeholder(tf.int32, [1])
	
	e1_e = tf.squeeze(tf.nn.embedding_lookup(ent_emb, h))
	r_e  = tf.squeeze(tf.nn.embedding_lookup(rel_emb, R))
	e2_e = tf.squeeze(tf.nn.embedding_lookup(ent_emb, l))

	dummy_img = tf.constant(np.zeros(n),dtype=tf.float32)

	e1_cmplx = tf.complex(e1_e, dummy_img)
	e2_cmplx = tf.complex(e2_e, dummy_img)
	ent_cmplx = tf.complex(ent_emb, tf.reshape(tf.tile(dummy_img, [num_ents]),[num_ents,n]))
	
	circ_correlation = tf.real(tf.ifft(tf.multiply(tf.conj(tf.fft(e1_cmplx)), tf.fft(ent_cmplx))))
	op = tf.sigmoid(tf.reduce_sum(tf.multiply(r_e, circ_correlation),axis=1))
	
	#circ_correlation_ = tf.real(tf.ifft(tf.multiply(tf.conj(tf.fft(ent_cmplx)), tf.fft(e2_cmplx))))
	#op_ = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(r_e, circ_correlation_),axis=1))

	with open('valid_tail_pred.txt','w') as fileout1:
		for t in validation_data:
			pos1 = sess.run(op ,feed_dict={h:[t[0]], R:[t[1]], l:[t[2]]})
			fileout1.write(' '.join([str(x) for x in pos1]) + '\n')
			#fileout2.write(' '.join([str(x) for x in pos2]) + '\n')
