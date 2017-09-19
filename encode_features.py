#!/usr/bin/env python3
import tensorflow as tf
from FeatureReader import FeatureReader
from AutoEncoder import AutoEncoder
import numpy as np
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

normalize_opts = [6, 6, 6, 6, 6, 6, 0, 1, 6, 6, 0, 5]


def main():
    cur_idx = 0
    display_steps = 100
    model_save_steps = 1000
    batch_size = 200

    train_examples, train_labels, test_examples, test_labels = FeatureReader(
        [cur_idx], normalize_opts[cur_idx], 'features').read()
    auto_encoder = AutoEncoder(np.shape(train_examples)[
                               1], [100, 100, 50, 100, 100])
    auto_encoder.build_graph()
    with tf.Session(graph=auto_encoder.get_graph()) as sess:
        auto_encoder.init_variables(sess)

        batch_examples, batch_labels = tf.train.shuffle_batch(
            [train_examples, train_labels], batch_size=batch_size, capacity=batch_size * 5, min_after_dequeue=batch_size * 4, enqueue_many=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_steps = 50000
        for i in range(num_steps):
            cur_examples = sess.run(batch_examples)
            loss = auto_encoder.train(sess, cur_examples)
            if i % display_steps == 0:
                print('Step: {}, Loss: {}'.format(i, loss))
            if i % model_save_steps == 0:
                auto_encoder.save_model(
                    sess, 'feature_{}_step_{}'.format(cur_idx, i))


if __name__ == '__main__':
    main()
