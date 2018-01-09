import collections
import numpy as np
import random
import math
import pickle
import tensorflow as tf
from common import build_word2vec_dataset, character_size


data, count, dictionary, reverse_dictionary = build_word2vec_dataset()

# 用来生成一个batch的训练样本

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer_ = collections.deque(maxlen=span)
    for _ in range(span):
        buffer_.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer_[skip_window]
            labels[i * num_skips + j, 0] = buffer_[target]
        buffer_.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch_size = 256
embedding_size = 256
skip_window = 2
num_skips = 4
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform(
            [character_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal(
            [character_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([character_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                         labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=character_size))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

    word2vec_saver = tf.train.Saver()
num_steps = 3000001
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0.0

    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            print("Average loss at step", step, ":", average_loss)
            average_loss = 0

        if step % 100000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s :" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    pickle.dump(final_embeddings, open('character_embeddings.pkl', 'wb'))
    print(final_embeddings)
    word2vec_saver.save(
        session, '/home/wuchao/paper/sentiment_analysis/word2vec/word2vec_model')

# 所有的配置参数
