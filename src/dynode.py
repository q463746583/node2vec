'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import tensorflow as tf
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
# import gensim
import os
import re
import time
import math
import collections
import random

data_index = 0
walk_index = 0
node_index = 0
node_to_index = dict()
index_to_node = dict()
emb = 0

def parse_args(fname_input, fname_output):
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default=fname_input,
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default=fname_output,
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
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)
    parser.add_argument('--gpus', type=int, default=2, help="amount of gpu")
    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=str, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, prev_emb_model, fname_model):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        if prev_emb_model == '0':
            # model = Word2Vec.load(namestring)
            walks = [list(map(str, walk)) for walk in walks]
            model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                             iter=args.iter)
            print(args.output)
            f = open(fname_model, "w+")
            model.wv.save_word2vec_format(args.output)
            model.save(fname_model)
        else:
            # print(prev_emb_model)
            # model = Word2Vec.load_word2vec_format()
            model = Word2Vec.load(prev_emb_model)
            walks = [list(map(str, walk)) for walk in walks]
            # model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
            # model.train(walks)
            model.build_vocab(walks, update=True)
            model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
            # model.wv.save_word2vec_format(args.output)
            print(args.output)
            f = open(fname_model, "w+")
            model.save(fname_model)

        return

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

def learn_embeddings_tensorflow(walks, nodes, prev_emb_model, fname_model):
    global index_to_node
    global node_to_index
    global node_index
    global emb
    # Parameters to learn

    if prev_emb_model == '0':
        for n in (nodes):
            index_to_node[node_index] = n
            node_to_index[n] = node_index
            node_index += 1
    else:
        for n in nodes:
            if n not in node_to_index.keys():
                index_to_node[node_index] = n
                node_to_index[n] = node_index
                node_index += 1
    print(node_to_index)
    print(index_to_node)
    vocabulary_size = len(index_to_node)
    window_size = args.window_size
    batch_size = 100
    num_sampled = 5
    embedding_size = args.dimensions

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])



        for i in range(2,6):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                # with tf.device('/cpu:0'):
                # with tf.device("/device:GPU:0"):
                    # Look up embeddings for inputs.
                _train_inputs = train_inputs[i * batch_size: (i+1) * batch_size]
                _train_labels = train_labels[i * batch_size: (i+1) * batch_size]

                with tf.name_scope('embeddings'):
                    if prev_emb_model == '0':
                        embeddings = tf.Variable(
                            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    else:
                        add_on_emb = tf.random_uniform([vocabulary_size - len(emb), embedding_size], -1.0, 1.0)
                        embeddings = tf.concat([emb, add_on_emb], 0)

                    embed = tf.nn.embedding_lookup(embeddings, _train_inputs)

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
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        #  Open a writer to write summaries.
        #  writer = tf.summary.FileWriter(log_dir, session.graph)

        #  We must initialize all variables before we use them.
        init.run()
        print('Initialized')
        average_loss = 0

        walks_data = []
        for w in walks:
            for n in w:
                walks_data.append(n)

        for step in range(args.iter):
            print(step)

            batch_inputs, batch_labels = generate_batch(batch_size, 1,
                                                        window_size, walks_data)
            # print("@@@@@@@@@@", batch_inputs)
            # print("**********", batch_labels)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
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

       
        # print(len(final_embeddings), len(final_embeddings[0]))
        # f = open(args.output, "w+")
        # for i in range(vocabulary_size):
        #     f.write(str(index_to_node[i]) + " " + ' '.join(str(e)
        #                                                  for e in final_embeddings.tolist()[i]) + "\n")
        emb = final_embeddings
        # save_path = saver.save(session, fname_model)
        
        # f.close()


def rebuild_model(walks, nodes, prev_emb_model,fname_model):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.25)
    config = tf.ConfigProto(gpu_options=gpu_options)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(prev_emb_model)
        saver.restore(sess, prev_emb_model)
        average_loss = 0

        walks_data = []
        for w in walks:
            for n in w:
                walks_data.append(n)

        for step in range(args.iter):

            batch_inputs, batch_labels = generate_batch(10, 1,
                                                        args.args.window_size, walks_data)
            # print("@@@@@@@@@@", batch_inputs)
            # print("**********", batch_labels)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, summary, loss_val = sess.run([optimizer, merged, loss],
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

        # print(final_embeddings)
        # print(len(final_embeddings), len(final_embeddings[0]))
        # f = open(args.output, "w+")
        #  f.write(str(vocab_size) + " " +  str(EMBEDDING_DIM) + "\n")
        # for i in range(len(nodes)):
        #     f.write(str(list(nodes)[i]) + " " + ' '.join(str(e)
        #                                                  for e in final_embeddings.tolist()[i]) + "\n")

        # save_path = saver.save(sess, fname_model)

        # f.close()


def main(args, prev_emb_model, fname_model):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    nx_G = read_graph()
    print(nx_G.nodes())
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)


    learn_embeddings_tensorflow(walks, nx_G.nodes(), prev_emb_model,fname_model)

    # if prev_emb_model == '0':
    # 	learn_embeddings_tensorflow(walks, nx_G.nodes(), prev_emb_model,fname_model)
    # else:
    # 	rebuild_model(walks, nx_G.nodes(), prev_emb_model,fname_model)




    # learn_embeddings_tensorflow(
    #     walks, nx_G.nodes(), prev_emb_model, fname_model)

# def main(args, prev_emb_model, fname_model):
# 	# type: (object) -> object
# 	'''
# 	Pipeline for representational learning for all nodes in a graph.
# 	'''
# 	nx_G = read_graph()
#     G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
# 	G.preprocess_transition_probs()
# 	walks = G.simulate_walks(args.num_walks, args.walk_length)
#     learn_embeddings_tensorflow(walks, nx_G.nodes(),prev_emb_model,fname_model)
    # learn_embeddings(walks,prev_emb_model,fname_model)


if __name__ == "__main__":
    DATA_DIR = './month/graph2'

    # DATA_DIR = 'data/darpa-big/training_edgeList_w'
    # DATA_DIR =  '../../../dataSetGeneration/Data/edge_list/'

    # This is for repeating the experiment
    # for i in range(0, 4):

    DATA_DIR1 = 'month/node_emb'
    DATA_DIR2 = 'month/model'
    # fnames = sorted(os.listdir(DATA_DIR))

    fnames = os.listdir(DATA_DIR)
    ordered_fnames = sorted(fnames, key=lambda x: (int(re.sub('\D', '', x)), x))

    # ordered_fnames = dict()
    global start_time
    start_time = time.time()

    # print (fnames)
    # print (ordered_fnames)
    # fnames.sort()
    '''
    count=0
    for curr_file in ordered_fnames:
        if count > 1:
            break
        print(curr_file)
        # with open(DATA_DIR + '/' + curr_file) as f:
        if count==0:
            # print(count)
            fname_input = DATA_DIR + '/' + curr_file
            fname_output = DATA_DIR1 + '/' + curr_file
            fname_model = DATA_DIR2 + '/' + curr_file
            prev_emb_model = '0'
            args = parse_args(fname_input, fname_output)
            main(args, prev_emb_model,fname_model)
            count += 1
        else:
            # print(count)
            # fname_model = DATA_DIR2 + '/' + curr_file
            prev_emb_model = fname_model
            # print(prev_emb_model)
            fname_input = DATA_DIR + '/' + curr_file
            fname_output = DATA_DIR1 + '/' + curr_file
            fname_model = DATA_DIR2 + '/' +  curr_file
            # print(fname_output)
            args = parse_args(fname_input,fname_output)
            main(args, prev_emb_model ,fname_model)
            count += 1
    '''
    for i, curr_file in enumerate(ordered_fnames):
        if i == 0:
            fname_input = DATA_DIR + '/' + curr_file
            fname_output = DATA_DIR1 + '/' + str(i) + '.emb'
            fname_model = DATA_DIR2 + '/' + str(i) + '.ckpt'
            prev_emb_model = '0'
            args = parse_args(fname_input, fname_output)
            main(args, prev_emb_model,fname_model)
        else:
            prev_emb_model = DATA_DIR2 + '/' + str(i-1) + '.ckpt'
            fname_input = DATA_DIR + '/' + curr_file
            fname_output = DATA_DIR1 + '/' + str(i) + '.emb'
            fname_model = DATA_DIR2 + '/' + str(i) + '.ckpt'
            args = parse_args(fname_input,fname_output)
            main(args, prev_emb_model ,fname_model)

    print("Elasped time", (time.time() - start_time))
