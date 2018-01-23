"""
    Define noise
"""
import tensorflow as tf

class BaseNet(object):

    def __init__(self, input_u=10, output_u=1):
        self.graph = tf.Graph()
        self.input_u = input_u
        self.output_u = output_u
        self.build()

    def build(self):
        """
        Simple 1 layer fully connected network to test the training module
        :param input_u: number of input neuron
        :param output_u: number of output neuron
        :return:
        """

        with self.graph.as_default():
            self.w = tf.get_variable(name='weights', shape=[self.input_u, self.output_u], dtype=tf.float32)
            self.b = tf.get_variable(name='bias', dtype=tf.float32, initializer=tf.zeros(shape=[1, self.output_u]))
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.input_u), name='X')
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.output_u), name='y')
            self.predict = tf.add(tf.matmul(self.X, self.w), self.b)

            self.loss = tf.reduce_sum(tf.square(self.predict - self.y))

            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            self.loss_summary = tf.summary.scalar("loss", self.loss)

# ==============================================================================================


class PairNet(object):

    """Build the architecture of pair based network"""

    def __init__(self, kernel=None, n_input=None, kernel_pre=None, kernel_post=None, ground_truth_init=1, reg_scale=[0.01, 0.01]):
        """
        Create and build the PairNet
        :param kernel: Kernel object
        :param n_input: input dimension
        :param kernel_pre: kernel used to convolve with pre-synaptic trains
        :param kernel_post: kernel used to convolve with post-synaptic trains
        :param reg_scale: l1, and l2 regularization strength
        """

        self.graph = tf.Graph()
        self.kernel = kernel
        self.n_input = n_input
        self.kernel_pre = kernel_pre
        self.kernel_post = kernel_post
        self.reg_scale = reg_scale
        self.ground_truth_init = ground_truth_init
        self.build()

    def build(self):
        """
        Build the architecture of the network
        :param: n_input, length of input
        :return:
        """
        with self.graph.as_default():

            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input, 2], name='inputs')
            self.x_pre, self.x_post = tf.unstack(self.inputs, axis=2)
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1], name='target')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            if self.ground_truth_init:  # Not in training model
                self.kernel_pre = tf.get_variable(shape=self.kernel_pre.shape, dtype=tf.float32, initializer=tf.constant_initializer(self.kernel_pre),
                                                  name='const_pre_kernel')
                self.kernel_post = tf.get_variable(shape=self.kernel_post.shape, dtype=tf.float32, initializer=tf.constant_initializer(self.kernel_post),
                                                   name='const_post_kernel')
            else:
                kernel_len = self.kernel.len_kernel
                self.kernel_pre = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1], name='pre_kernel')
                self.kernel_post = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1], name='post_kernel')

            self.bias = tf.Variable(0, dtype=tf.float32, name='bias')

            self.y_pre = self.conv_1d(data=self.x_pre, kernel=self.kernel_pre)
            self.y_post = self.conv_1d(data=self.x_post, kernel=self.kernel_post)

            self.prediction = tf.matmul(a=self.y_pre, b=self.y_post, transpose_a=True) + self.bias

            self.loss = tf.reduce_sum(tf.square(self.prediction - self.target))

            self.alpha_l1 = self.reg_scale[0]
            self.alpha_l2 = self.reg_scale[1]

            self.total = self.loss + self.regularization()

    def conv_1d(self, data=None, kernel=None):
        """
        Wrapper function for performing 1d convolution
        :param data: data to convolve on
        :param kernel: filter to perform the convolution
        :return:
        """
        # Reshape the input data to the shape:[batch, in_width, in_channels]
        x = tf.expand_dims(input=data, axis=2)
        # Reshape kernel to the shape: [filter_width, in_channels, out_channels]
        kernel = tf.expand_dims(input=kernel, axis=2)
        #y = tf.squeeze(tf.nn.conv1d(value=x, filters=kernel, stride=1, padding='SAME'))
        y = tf.nn.conv1d(value=x, filters=kernel, stride=1, padding='SAME')

        return y

    def regularization(self):

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.alpha_l1, scope=None)
        l1_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [self.kernel_pre, self.kernel_post])

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.alpha_l2, scope=None)
        l2_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, [self.kernel_pre, self.kernel_post])

        return l1_penalty + l2_penalty


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
