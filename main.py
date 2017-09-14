#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import collections
from FeatureReader import FeatureReader


FLAGS = tf.app.flags.FLAGS


def main(_):

    train_examples, train_labels, test_examples, test_labels = FeatureReader(
        [1, 3, 8], 'features').read()
    print('train_examples: ', np.shape(train_examples))
    print('train_labels: ', np.shape(train_labels))
    print('test_examples: ', np.shape(test_examples))
    print('test_labels: ', np.shape(test_labels))

    input_dim = np.shape(train_examples)[1]
    output_dim = 1
    batch_size = 100
    num_preprocess_threads = 1
    min_queue_examples = 256

    learning_rate = 0.3
    layer_conf = [30, 50, 60]
    layer_matrix = []
    bias_matrix = []
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=[
                                batch_size, input_dim])
        labels = tf.placeholder(dtype=tf.float32, shape=[
                                batch_size, output_dim])
        cur = inputs
        for i in range(len(layer_conf)):
            if i == 0:
                mat = tf.Variable(tf.random_normal([input_dim, layer_conf[i]]))
            else:
                mat = tf.Variable(tf.random_normal(
                    [layer_conf[i - 1], layer_conf[i]]))
            bias = tf.Variable(tf.random_normal([layer_conf[i]]))
            layer_matrix.append(mat)
            bias_matrix.append(bias)
            cur = tf.add(tf.matmul(cur, mat), bias)

        mat = tf.Variable(tf.random_normal([layer_conf[-1], output_dim]))
        bias = tf.Variable(tf.random_normal([output_dim]))
        layer_matrix.append(mat)
        bias_matrix.append(bias)
        logits = tf.add(tf.matmul(cur, mat), bias)
        loss_op = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        init = tf.initialize_all_variables()

    num_steps = 50000
    with tf.Session(graph=graph) as sess:
        sess.run(init)

        batch_examples = tf.train.batch([train_examples], batch_size=batch_size, num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 3 * batch_size, enqueue_many=True, allow_smaller_final_batch=True)
        batch_labels = tf.train.batch([train_labels], batch_size=batch_size, num_threads=num_preprocess_threads,
                                      capacity=min_queue_examples + 3 * batch_size, enqueue_many=True, allow_smaller_final_batch=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_steps):
            print(i)
            cur_examples = sess.run(batch_examples)
            cur_labels = sess.run(batch_labels)
            print(np.shape(cur_examples))
            print(np.shape(cur_labels))

            loss, _ = sess.run([loss_op, train_op], feed_dict={
                inputs: cur_examples, labels: cur_labels.reshape((-1, 1))})


if __name__ == '__main__':
    tf.app.run()
