#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import collections
from FeatureReader import FeatureReader
from tensorflow.examples.tutorials.mnist import input_data


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

FLAGS = tf.app.flags.FLAGS

def main(_):

    train_examples, train_labels, test_examples, test_labels = FeatureReader(range(12), [6, 6, 6, 6, 6, 6, 0, 1, 6, 6, 0, 5], 'features').read()
    print('train_examples: ', np.shape(train_examples))
    print('train_labels: ', np.shape(train_labels))
    print('test_examples: ', np.shape(test_examples))
    print('test_labels: ', np.shape(test_labels))

    # MNIST 
    # input_dim = 784
    # output_dim = 10

    input_dim = np.shape(train_examples)[1]
    output_dim = 2
    batch_size = 10
    num_preprocess_threads = 1
    min_queue_examples = 256

    learning_rate = 0.5
    layer_conf = [50, 50, 50, 50, 50, 50, 50]
    layer_matrix = []
    bias_matrix = []
    prob_matrix = []
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=[
                                None, input_dim])
        labels = tf.placeholder(dtype=tf.int32, shape=[None])
        one_hot_labels = tf.one_hot(labels, depth=output_dim) 

        prob_matrix.append(inputs)
        for i in range(len(layer_conf)):
            if i == 0:
                mat = tf.Variable(tf.random_normal([input_dim, layer_conf[i]]))
            else:
                mat = tf.Variable(tf.random_normal(
                    [layer_conf[i - 1], layer_conf[i]]))
            bias = tf.Variable(tf.random_normal([layer_conf[i]]))
            layer_matrix.append(mat)
            bias_matrix.append(bias)
            prob = tf.nn.relu(tf.add(tf.matmul(prob_matrix[-1], mat), bias))
            prob_matrix.append(prob)

        mat = tf.Variable(tf.random_normal([layer_conf[-1], output_dim]))
        bias = tf.Variable(tf.random_normal([output_dim]))
        layer_matrix.append(mat)
        bias_matrix.append(bias)

        logits = tf.add(tf.matmul(prob_matrix[-1], mat), bias)
        prob = tf.nn.softmax(logits)
        predicts = tf.argmax(prob, 1)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels))
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss_op)

        # Test accuracy
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predicts)

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()


    num_steps = 500000
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        sess.run(init_local)

        batch_examples, batch_labels = tf.train.shuffle_batch([train_examples, train_labels], batch_size=batch_size, capacity=10000, min_after_dequeue=5000, enqueue_many=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_steps):
            # cur_examples, cur_labels = mnist.train.next_batch(batch_size)
            cur_examples, cur_labels = sess.run([batch_examples, batch_labels])

            _, loss = sess.run([train_op, loss_op], feed_dict={
                    inputs: cur_examples, labels: cur_labels})

            if i % 100 == 0:
                print('loss: ', loss)
            if i % 1000 == 0:
                _, cur_acc = sess.run([acc_op, acc], feed_dict={inputs: test_examples, labels: test_labels})
                print('# accuracy: ', cur_acc)
            


if __name__ == '__main__':
    tf.app.run()
