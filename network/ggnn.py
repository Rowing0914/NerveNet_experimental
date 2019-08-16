import tensorflow as tf


class Propagator(tf.keras.Model):
    """ LSTM Mechanism Cell """

    def __init__(self, state_dim):
        super(Propagator, self).__init__()
        self.reset_gate = tf.keras.layers.Dense(state_dim, activation="sigmoid")
        self.update_gate = tf.keras.layers.Dense(state_dim, activation="sigmoid")
        self.transform = tf.keras.layers.Dense(state_dim, activation="sigmoid")

    def call(self, inputs, prev_state):
        r = self.reset_gate(inputs)
        z = self.update_gate(inputs)
        h_hat = self.transform(inputs)
        output = (1 - z) * prev_state * r + z * h_hat
        return output


class GGNNBlock(tf.keras.Model):
    def __init__(self, adjacency_matrix, state_dim):
        super(GGNNBlock, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.dense = tf.keras.layers.Dense(state_dim, use_bias=True)
        self.lstm_cell = tf.compat.v1.keras.layers.CuDNNLSTM(state_dim, stateful=True, return_sequences=True)
        self.propagate = Propagator(state_dim=state_dim)

    def call(self, features, training=None, mask=None):
        features = self.dense(features)
        for _ in range(2):
            messages = tf.linalg.matmul(self.adjacency_matrix, features)
            # NOTE: since the lstm_cell is not working well, I've decided to implement it myself
            # features = self.lstm_cell(messages)
            features = self.propagate(messages, features)
        features = tf.keras.activations.relu(features)
        return features


class GGNN(tf.keras.Model):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, adjacency_matrix, num_node_feature):
        super(GGNN, self).__init__()
        self._ggnn = GGNNBlock(adjacency_matrix, num_node_feature)
        self._readout = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, features, training=None, mask=None):
        output = self._ggnn(features)
        action = self._readout(output)
        return action


class MLP(tf.keras.Model):
    def __init__(self, dims, activations):
        super(MLP, self).__init__()
        self.num_layers = len(dims)-1
        self._summary = ""
        for i in range(self.num_layers):
            setattr(self, "layer_{}".format(i), tf.keras.layers.Dense(dims[i], activation=activations[i]))
            self._summary += "| layer: {} | dim: {} | activation: {} |\n".format(i+1, str(dims[i]), activations[i])

    def call(self, inputs):
        hidden = inputs
        for i in range(self.num_layers):
            hidden = getattr(self, "layer_{}".format(i))(hidden)
        return hidden

    def __str__(self):
        return self._summary

