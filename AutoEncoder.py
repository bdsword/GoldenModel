import tensorflow as tf


class AutoEncoder:
    def __init__(self, input_dim, layers_conf, learning_rate=0.5):
        if len(layers_conf) % 2 == 0:
            raise Exception('Number of layer should be an odd number.')
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.layers_conf = layers_conf
        self.learning_rate = learning_rate

        self.layers_matrix = []
        self.biases_matrix = []
        self._layers_outputs = []

        # For decode only purpose
        self._decode_layers_outputs = []

        self._graph = None

    def build_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            code_layer_idx = len(self.layers_conf) // 2

            self.inputs = tf.placeholder(dtype=tf.float32, shape=[
                None, self.input_dim])
            self._layers_outputs.append(self.inputs)

            # For decode only purpose
            self.input_codes = tf.placeholder(dtype=tf.float32, shape=[
                None, self.layers_conf[code_layer_idx]])
            self._decode_layers_outputs.append(self.input_codes)

            for i in range(code_layer_idx + 1):
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
                    tf.add(tf.matmul(self._layers_outputs[-1], mat), bias))
                self._layers_outputs.append(prob)

            output_bias = tf.Variable(tf.random_normal([self.input_dim]))
            self.biases_matrix.append(output_bias)
            for i in range(code_layer_idx + 1):
                mat = self.layers_matrix[code_layer_idx - i]
                bias = self.biases_matrix[code_layer_idx - i - 1]
                prob = tf.nn.relu(
                    tf.add(tf.matmul(self._layers_outputs[-1], tf.transpose(mat)), bias))
                self._layers_outputs.append(prob)

                # For decode only purpose
                prob = tf.nn.relu(
                    tf.add(tf.matmul(self._decode_layers_outputs[-1], tf.transpose(mat)), bias))
                self._decode_layers_outputs.append(prob)

            y_pred = self._layers_outputs[-1]
            y_true = self.inputs
            self.loss_op = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss_op)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

    def init_variables(self, session):
        session.run(self.init)

    def get_graph(self):
        if self._graph is None:
            raise Exception('Please build the graph first.')
        return self._graph

    def train(self, sess, batch_examples):
        _, loss = sess.run([self.train_op, self.loss_op], feed_dict={
            self.inputs: batch_examples})
        return loss

    def encode(self, session, examples):
        code_layer_idx = len(self.layers_conf) // 2 + 1
        return session.run(self._layers_outputs[code_layer_idx], feed_dict={
            self.inputs: examples})

    def decode(self, session, examples):
        return session.run(self._decode_layers_outputs[-1], feed_dict={self.input_codes: examples})

    def save_model(self, session, filename):
        self.saver.save(session, filename)

    def restore_model(self, session, model_root, filename):
        self.saver = tf.train.import_meta_graph(filename)
        self.saver.restore(session, tf.train.latest_checkpoint(model_root))
