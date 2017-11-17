import tensorflow as tf
import numpy as np
import ctypes
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

lib = ctypes.cdll.LoadLibrary('./batchGen.so')

num_ents = 14951
num_rels = 1345
batch_size = 10000
n = 100
nepoch = 10000
learning_rate = 0.001
dropout_prob = 0.5
reg_param = 0.0

def read_data(filename):
	train_triples = []
	with open(filename,'r') as filein:
		for line in filein:
			train_triples.append([int(x.strip()) for x in line.split()])
	return train_triples

with tf.device('/gpu:0'):

	regularizer = tf.contrib.layers.l2_regularizer(reg_param)
	ent_emb = tf.get_variable(initializer=tf.random_uniform(shape=[num_ents, n], minval=-1.0, maxval=1.0, dtype=tf.float32,seed=1),name='ent_emb',regularizer=regularizer)
	rel_emb = tf.get_variable(initializer=tf.random_uniform(shape=[num_rels, n], minval=-1.0, maxval=1.0, dtype=tf.float32,seed=1),name='rel_emb',regularizer=regularizer)
	global_step = tf.Variable(0, name='global_step', trainable=False)

	pos_head = tf.placeholder(tf.int32, [batch_size])
	pos_tail = tf.placeholder(tf.int32, [batch_size])
	rel      = tf.placeholder(tf.int32, [batch_size])
	neg_head = tf.placeholder(tf.int32, [batch_size])
	neg_tail = tf.placeholder(tf.int32, [batch_size])

	pos_head_e = tf.nn.embedding_lookup(ent_emb, pos_head)
	pos_tail_e = tf.nn.embedding_lookup(ent_emb, pos_tail)
	rel_e      = tf.nn.embedding_lookup(rel_emb, rel)
	neg_head_e = tf.nn.embedding_lookup(ent_emb, neg_head)
	neg_tail_e = tf.nn.embedding_lookup(ent_emb, neg_tail)

	pos_labels = tf.constant(np.ones(batch_size),dtype=tf.float32)
	neg_labels = tf.constant(np.zeros(batch_size),dtype=tf.float32)
	#neg_labels = tf.constant(-1 * np.ones(batch_size),dtype=tf.float32)

	labels = tf.concat([pos_labels, neg_labels],axis=0)

	input_l = tf.concat([tf.concat([pos_head_e,rel_e,pos_tail_e],axis=1), tf.concat([neg_head_e, rel_e, neg_tail_e],axis=1)] , axis=0) 
	#input_l = tf.concat([tf.concat([pos_head_e + rel_e, pos_tail_e],axis=1), tf.concat([neg_head_e + rel_e, neg_tail_e],axis=1)] , axis=0)

	Wh1 = tf.get_variable(shape=[3*n,10*n],initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=1), name='Wh1') 

	Hl1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input_l, Wh1)), dropout_prob)
	
	Wh2 = tf.get_variable(shape=[10*n, 1],initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=1), name='Wh2')
	bias2 = tf.Variable(tf.zeros([1]), name='bias2')

	logits = tf.squeeze(tf.matmul(Hl1, Wh2))  
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	#loss = tf.reduce_sum(tf.log(1 + tf.exp(-1 * labels * logits)))
	#loss = tf.reduce_sum(tf.exp(-1 * labels * logits))
	
	reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss += sum(reg_term)

	tf.summary.scalar('Cross-Entropy loss, ER-MLP',loss)
	
	merged = tf.summary.merge_all()
	saver = tf.train.Saver(max_to_keep=100)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

	ent_normalizer = tf.assign(ent_emb, tf.nn.l2_normalize(ent_emb,dim=1))
	rel_normalizer = tf.assign(rel_emb, tf.nn.l2_normalize(rel_emb,dim=1))

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
		sess.run([ent_normalizer, rel_normalizer])
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
