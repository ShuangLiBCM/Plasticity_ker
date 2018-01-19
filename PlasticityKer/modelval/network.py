"""
    Create network classes
"""
import tensorflow as tf

class Network(object):

    def __init__(self, kernel=[]):   # If ground truth kernel is given ,it will be used.
        self.graph = tf.Graph()
        self.kernel = kernel

    def build(self, input_u, output_u):

        with tf.variable_scope("Network", reuse=True):
            self.w = tf.get_variable(name='weights', shape=[input_u, output_u], dtype=tf.float32)
            self.b = tf.get_variable(name='bias', dtype=tf.float32, initializer=tf.zeros([1, output_u]))
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, input_u), name='X')

            self.predict = tf.matmul(self.X, self.w) + self.b

        return self.predict

    def loss(self, target):
        """
            Deal with generating the prediction
        """
        self.target = target
        pass

    def train(self):
        """
            Deal with the training.
        """
        pass

    def batch(self):
        """
            Deal with the data.
        """
        pass


class PairNet(Network):

    def __init__(self, kernel=[]):
        super(Network, self).__init__(kernel=[])

    def build(self, w_pre, w_post, b_pre, b_post):
        """
        Build the architecture of the network
        :return:
        """
        # self.w_pre = tf.get_variable(name='w_pre', shape=w_pre.shape, )
        # self.w_post = tf.get_variable(name='w_pre', shape=w_pre.shape, )
        # self.b = tf.get_variable()
        # self.input = tf.placeholder()
        # self.output = tf.placeholder()
        pass


class TripNet(Network):

    def __init__(self, kernel=[]):
        super(Network, self).__init__(kernel=[])

    def build(self, w_pre, w_post, b_pre, b_post):
        """
        Build the architecture of the network
        :return:
        # """
        # self.w_pre = tf.get_variable(name='w_pre', shape=w_pre.shape, )
        # self.w_post = tf.get_variable(name='w_pre', shape=w_pre.shape, )
        # self.b = tf.get_variable()
        # self.input = tf.placeholder()
        # self.output = tf.placeholder()
        pass

