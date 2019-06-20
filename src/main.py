'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import os
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import tensorflow as tf
import math
import collections
import random
import time
import random

data_index = 0
walk_index = 0
node_to_index = dict()
index_to_node = dict()


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/month_1_graph.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/test3.emb',
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
    parser.add_argument('--unweighted', dest='unweighted',
                        action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected',
                        action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int,
                             create_using=nx.DiGraph())
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
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return


def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index - 1] = 1
    return temp


def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    global walk_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(
        maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    # if data_index - skip_window < 0:
    #   buffer.extend(data[0 : data_index])
    # else:
    #   buffer.exntend(data[data_index-skip_window : data_index])
    # if data_index + skip_window + 1 >= args.walk_length:
    #   buffer.exnted(data[data_index + 1 : args.walk_length - 1])
    # else:
    #   buffer.extned(data[data_index + 1 : data_index + skip_window + 1])

    data_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = node_to_index[buffer[skip_window]]
            labels[i * num_skips + j, 0] = node_to_index[buffer[context_word]]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def learn_embeddings_tensorflow(walks, nodes):
    global index_to_node
    global node_to_index
    # Parameters to learn

    for index, n in enumerate(nodes):
        index_to_node[index] = n
        node_to_index[n] = index

    vocabulary_size = len(nodes)
    window_size = args.window_size
    batch_size = 10
    num_sampled = 5
    embedding_size = args.dimensions

    graph = tf.Graph()
    with graph.as_default():
        tower_grads = []
        reuse_vars = False
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    
        for i in range(1,7):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                # with tf.device('/cpu:0'):
                # with tf.device("/device:GPU:0"):
                    # Look up embeddings for inputs.
                _train_inputs = train_inputs[i * batch_size: (i+1) * batch_size]
                _train_labels = train_labels[i * batch_size: (i+1) * batch_size]
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, _train_inputs)

                # for d in ['/device:GPU:6', '/device:GPU:7', '/device:GPU:4', '/device:GPU:5', '/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
                #      with tf.device(d):
                #         # Look up embeddings for inputs.
                #         with tf.name_scope('embeddings'):
                #             embeddings = tf.Variable(
                #                 tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                #             embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal([vocabulary_size, embedding_size],
                                            stddev=1.0 / math.sqrt(embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

                    # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                # Explanation of the meaning of NCE loss:
                #   http://mccormickml.com/2016/04/19/waord2vec-tutorial-the-skip-gram-model/
                
                # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=_train_labels,
                            inputs=embed,
                            num_sampled=num_sampled,
                            num_classes=vocabulary_size))


                # Construct the SGD optimizer using a learning rate of 1.0.
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.GradientDescentOptimizer(
                        1.0).minimize(loss, colocate_gradients_with_ops=True)


        # Compute the cosine similarity between minibatch examples and all
        # embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
    # with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    
        # Open a writer to write summaries.
        #  writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')
        average_loss = 0

        walks_data = []
        for w in walks:
            for n in w:
                walks_data.append(n)

        for step in range(args.iter):
            # i = random.randint(1,7)
            # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            # print("&&&&&&&&&&&&", i)
            batch_inputs, batch_labels = generate_batch(batch_size, 1,
                                                        window_size, walks_data)
            print("@@@@@@@@@@", batch_inputs)
            print("**********", batch_labels)
            feed_dict = {train_inputs: batch_inputs,
                            train_labels: batch_labels}
            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, loss_val = session.run([optimizer, loss],
                                                feed_dict=feed_dict,
                                                run_metadata=run_metadata)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000
                    # batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

        final_embeddings = normalized_embeddings.eval()
        print("Elasped time", (time.time() - start_time))
        print(final_embeddings)
        print(len(final_embeddings), len(final_embeddings[0]))
        f = open(args.output, "w+")
        #  f.write(str(vocab_size) + " " +  str(EMBEDDING_DIM) + "\n")
        for i in range(vocabulary_size):
            f.write(str(list(nodes)[i]) + " " + ' '.join(str(e)
                                                            for e in final_embeddings.tolist()[i]) + "\n")
        f.close()


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    global start_time
    start_time = time.time()
    nx_G = read_graph()
    print(nx_G.nodes())
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings_tensorflow(walks, nx_G.nodes())
    # learn_embeddings(walks)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "04:00.0"
    args = parse_args()
    main(args)
