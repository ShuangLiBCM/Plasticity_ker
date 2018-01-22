"""
    Define noise
"""
import tensorflow as tf

class BaseNet(object):

    def __init__(self):
        self.graph = tf.Graph()
        self.build()

    def build(self, input_u=10, output_u=1):
        """
        Simple 1 layer fully connected network to test the training module
        :param input_u: number of input neuron
        :param output_u: number of output neuron
        :return:
        """

        with self.graph.as_default():

            self.w = tf.get_variable(name='weights', shape=[input_u, output_u], dtype=tf.float32)
            self.b = tf.get_variable(name='bias', dtype=tf.float32, initializer=tf.zeros(shape=[1, output_u]))
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, input_u), name='X')
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, output_u), name='y')
            self.predict = tf.add(tf.matmul(self.X, self.w), self.b)

            self.loss = tf.reduce_sum(tf.square(self.predict - self.y))

            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            self.loss_summary = tf.summary.scalar("loss", self.loss)

class PairNet(object):
    """Build the architecture of pair based network"""

    def __init__(self):
        pass

    def build(self):
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


class TripNet(object):

    def __init__(self):
        pass

    def build(self):
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

