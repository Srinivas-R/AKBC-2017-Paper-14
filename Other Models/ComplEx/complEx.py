import tensorflow as tf
import numpy as np
import ctypes
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

lib = ctypes.cdll.LoadLibrary('./batchGen.so')

num_ents = 75043
num_rels = 13
batch_size = 50000
n = 200
nepoch = 10000
learning_rate = 0.001
reg_param = 0.001

def read_data(filename):
	train_triples = []
	with open(filename,'r') as filein:
		for line in filein:
			train_triples.append([int(x.strip()) for x in line.split()])
	return train_triples

with tf.device('/gpu:1'):

	regularizer = tf.contrib.layers.l2_regularizer(reg_param)
	ent_real = tf.get_variable(initializer=tf.random_uniform(shape=[num_ents, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='ent_real', regularizer=regularizer)
	ent_img = tf.get_variable(initializer=tf.random_uniform(shape=[num_ents, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='ent_img', regularizer=regularizer)
	rel_real = tf.get_variable(initializer=tf.random_uniform(shape=[num_rels, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_real', regularizer=regularizer)
	rel_img = tf.get_variable(initializer=tf.random_uniform(shape=[num_rels, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_img', regularizer=regularizer)

	global_step = tf.Variable(0, name='global_step', trainable=False)

	pos_head = tf.placeholder(tf.int32, [batch_size])
	pos_tail = tf.placeholder(tf.int32, [batch_size])
	rel      = tf.placeholder(tf.int32, [batch_size])
	neg_head = tf.placeholder(tf.int32, [batch_size])
	neg_tail = tf.placeholder(tf.int32, [batch_size])
	
	pos_head_real_e = tf.nn.embedding_lookup(ent_real, pos_head)
	pos_head_img_e  = tf.nn.embedding_lookup(ent_img, pos_head)
	pos_tail_real_e = tf.nn.embedding_lookup(ent_real, pos_tail)
	pos_tail_img_e  = tf.nn.embedding_lookup(ent_img, pos_tail)
	rel_real_e      = tf.nn.embedding_lookup(rel_real, rel)
	rel_img_e       = tf.nn.embedding_lookup(rel_img, rel)
	neg_head_real_e = tf.nn.embedding_lookup(ent_real, neg_head)
	neg_head_img_e  = tf.nn.embedding_lookup(ent_img, neg_head)
	neg_tail_real_e = tf.nn.embedding_lookup(ent_real, neg_tail)
	neg_tail_img_e  = tf.nn.embedding_lookup(ent_img, neg_tail)

	pos_labels = tf.constant(np.ones(batch_size),dtype=tf.float32)
	#neg_labels = tf.constant(np.zeros(num_neg_samples * batch_size),dtype=tf.float32)
	neg_labels = tf.constant(-1 * np.ones(batch_size),dtype=tf.float32)

	labels = tf.concat([pos_labels, neg_labels], axis=0)
	
	score1_pos = tf.reduce_sum(tf.multiply(tf.multiply(rel_real_e, pos_head_real_e),pos_tail_real_e), axis=1)
	score2_pos = tf.reduce_sum(tf.multiply(tf.multiply(rel_real_e, pos_head_img_e),pos_tail_img_e), axis=1)
	score3_pos = tf.reduce_sum(tf.multiply(tf.multiply(rel_img_e, pos_head_real_e),pos_tail_img_e), axis=1)
	score4_pos = tf.reduce_sum(tf.multiply(tf.multiply(rel_img_e, pos_head_img_e),pos_tail_real_e), axis=1)
	tot_score_pos = score1_pos + score2_pos + score3_pos - score4_pos

	score1_neg = tf.reduce_sum(tf.multiply(tf.multiply(rel_real_e, neg_head_real_e),neg_tail_real_e), axis=1)
	score2_neg = tf.reduce_sum(tf.multiply(tf.multiply(rel_real_e, neg_head_img_e),neg_tail_img_e), axis=1)
	score3_neg = tf.reduce_sum(tf.multiply(tf.multiply(rel_img_e, neg_head_real_e),neg_tail_img_e), axis=1)
	score4_neg = tf.reduce_sum(tf.multiply(tf.multiply(rel_img_e, neg_head_img_e),neg_tail_real_e), axis=1)
	tot_score_neg = score1_neg + score2_neg + score3_neg - score4_neg

	logits = tf.concat([tot_score_pos, tot_score_neg], axis=0)

	#loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
	#loss = tf.reduce_sum(tf.exp(-1 * labels * logits))
	loss = tf.reduce_sum(tf.log(1 + tf.exp(-1 * labels * logits)))

	reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss += sum(reg_term)
	
	tf.summary.scalar('ComplEx negative log loss',loss)
	merged = tf.summary.merge_all()
	saver = tf.train.Saver(max_to_keep=100)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
	#grads_and_vars = optimizer.compute_gradients(loss)
	#opt_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	# ent_real_normalizer = tf.assign(ent_real, tf.nn.l2_normalize(ent_real,dim=1))
	# rel_real_normalizer = tf.assign(rel_real, tf.nn.l2_normalize(rel_real,dim=1))
	# ent_img_normalizer = tf.assign(ent_img, tf.nn.l2_normalize(ent_img,dim=1))
	# rel_img_normalizer = tf.assign(rel_img, tf.nn.l2_normalize(rel_img,dim=1))

data = np.array(read_data('./data/train.txt'))
nbatches = len(data) // batch_size

ph = np.zeros(batch_size, dtype = np.int32)
pt = np.zeros(batch_size, dtype = np.int32)
r = np.zeros(batch_size, dtype = np.int32)
nh = np.zeros(batch_size, dtype = np.int32)
nt = np.zeros(batch_size, dtype = np.int32)
	 		
ph_addr = ph.__array_interface__['data'][0]
pt_addr = pt.__array_interface__['data'][0]
r_addr = r.__array_interface__['data'][0]
nh_addr = nh.__array_interface__['data'][0]
nt_addr = nt.__array_interface__['data'][0]

lib.init()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('logs', sess.graph)
	for _ in range(nepoch):
		epoch_loss = 0.0
		for __ in range(nbatches):
			lib.getBatch(ph_addr, pt_addr, r_addr, nh_addr, nt_addr, batch_size, 0)
			feed_dict = {pos_head : ph,
						 pos_tail : pt, 
						 rel 	  : r,
						 neg_head : nh,
					 	 neg_tail : nt}

			l,summary,a= sess.run([loss,merged,optimizer],feed_dict)
			writer.add_summary(summary, tf.train.global_step(sess, global_step))

			epoch_loss += l
		print('Epoch {}\tLoss {}'.format(_,epoch_loss))
		if (_+1) % 1000 == 0:
			saver.save(sess, 'logs/model.vec', global_step=global_step)
	print('Finished training the model')
