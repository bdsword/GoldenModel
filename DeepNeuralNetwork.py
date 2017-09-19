#!/usr/bin/env python3

import tensorflow as tf


class DeepNeuralNetwork:
    def __init__(self, input_dim, output_dim, layers_conf, learning_rate=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_conf = layers_conf
        self.learning_rate = learning_rate

        self.layers_matrix = []
        self.biases_matrix = []
        self.probs_matrix = []

        self._graph = None

    def build_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[
                None, self.input_dim])
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None])
            one_hot_labels = tf.one_hot(self.labels, depth=self.output_dim)

            self.probs_matrix.append(self.inputs)
            for i in range(len(self.layers_conf)):
                if i == 0:
                    mat = tf.Variable(tf.random_normal(
                        [self.input_dim, self.layers_conf[i]]))
                else:
                    mat = tf.Variable(tf.random_normal(
                        [self.layers_conf[i - 1], self.layers_conf[i]]))
                bias = tf.Variable(tf.random_normal([self.layers_conf[i]]))
                self.layers_matrix.append(mat)
                self.biases_matrix.append(bias)
                prob = tf.nn.relu(
                    tf.add(tf.matmul(self.probs_matrix[-1], mat), bias))
                self.probs_matrix.append(prob)

            mat = tf.Variable(tf.random_normal(
                [self.layers_conf[-1], self.output_dim]))
            bias = tf.Variable(tf.random_normal([self.output_dim]))
            self.layers_matrix.append(mat)
            self.biases_matrix.append(bias)

            logits = tf.add(tf.matmul(self.probs_matrix[-1], mat), bias)
            prob = tf.nn.softmax(logits)
            self.predicts = tf.argmax(prob, 1)
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=one_hot_labels))
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss_op)

            # Test accuracy
            self.acc, self.acc_op = tf.metrics.accuracy(
                labels=self.labels, predictions=self.predicts)

            self.init = tf.global_variables_initializer()
            self.init_local = tf.local_variables_initializer()

    def init_variables(self, session):
        session.run(self.init)
        session.run(self.init_local)

    def get_graph(self):
        if self._graph is None:
            raise Exception('Please build the graph first.')
        return self._graph

    def get_accuracy(self, sess, test_examples, test_labels):
        _, cur_acc = sess.run([self.acc_op, self.acc], feed_dict={
                              self.inputs: test_examples, self.labels: test_labels})
        return cur_acc

    def train(self, sess, batch_examples, batch_labels):
        _, loss = sess.run([self.train_op, self.loss_op], feed_dict={
            self.inputs: batch_examples, self.labels: batch_labels})
        return loss
