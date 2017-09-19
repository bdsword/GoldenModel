from FeatureReader import FeatureReader
from DeepNeuralNetwork import DeepNeuralNetwork
import tensorflow as tf
import numpy as np
import sys
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

FLAGS = tf.app.flags.FLAGS

def read_data():
    train_examples, train_labels, test_examples, test_labels = FeatureReader(range(12), [6, 6, 6, 6, 6, 6, 0, 1, 6, 6, 0, 5], 'features').read()
    print('train_examples: ', np.shape(train_examples))
    print('train_labels: ', np.shape(train_labels))
    print('test_examples: ', np.shape(test_examples))
    print('test_labels: ', np.shape(test_labels))
    return train_examples, train_labels, test_examples, test_labels

def main(_):
    train_examples, train_labels, test_examples, test_labels = read_data()
    dnn = DeepNeuralNetwork(np.shape(train_examples)[1], 2, [1024, 512, 256, 128, 32])
    dnn.build_graph()

    batch_size = 200

    with tf.Session(graph=dnn.get_graph()) as sess:
        dnn.init_variables(sess)

        batch_examples, batch_labels = tf.train.shuffle_batch([train_examples, train_labels], batch_size=batch_size, capacity=batch_size * 5, min_after_dequeue=batch_size * 4, enqueue_many=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_steps = 500000
        for i in range(num_steps):
            cur_examples, cur_labels = sess.run([batch_examples, batch_labels])
            loss = dnn.train(sess, cur_examples, cur_labels)

            if i % 100 == 0:
                print('loss: ', loss)
            if i % 1000 == 0:
                acc = dnn.get_accuracy(sess, test_examples, test_labels)
                print('# accuracy: ', acc)
                dnn.save_model(sess, 'model/dnn_step_{}'.format(i))

if __name__ == '__main__':
    tf.app.run()

