'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import tensorflow as tf

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/test.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	# for w in walks: print(w[0])
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index - 1] = 1
    return temp

def learn_embeddings_modify(walks, vocab_size):
	'''
	Implement the embedding method via tensorflow 
	'''
	data = []
	# walks = [list(map(str, walk)) for walk in walks]
	for walk in walks:
		for index, node in enumerate(walk):
			for nb_node in walk[max(index - args.window_size, 0) : min(index + args.window_size, len(walk)) + 1]:
				# if nb_node != node:
				# 	data.append([node, nb_node])
				data.append([node, nb_node])
	x_train = []
	y_train = []
	
	for data_word in data:
		x_train.append(to_one_hot(data_word[0] , vocab_size))
		y_train.append(to_one_hot(data_word[1] , vocab_size))

	# convert them to numpy arrays
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)

	print(x_train.shape, y_train.shape)
	
	# making placeholders for x_train and y_train
	x = tf.placeholder(tf.float32, shape=(None, vocab_size))
	y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))
	

	EMBEDDING_DIM = args.dimensions 

	W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
	b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
	hidden_representation = tf.add(tf.matmul(x,W1), b1)

	W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
	b2 = tf.Variable(tf.random_normal([vocab_size]))
	prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init) #make sure you do this!
	# define the loss function:
	cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
	# define the training step:
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
	n_iters = args.iter
	# train for n_iter iterations
	for _ in range(n_iters):
		sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
		# print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
	print(sess.run(W1 + b1))
			
	
	 


def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings_modify(walks, len(nx_G.nodes()))
	#learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
