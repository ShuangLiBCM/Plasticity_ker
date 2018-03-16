"""
    Define noise
"""
import tensorflow as tf
import numpy as np
from scipy import signal


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

    def __init__(self, kernel=None, n_input=None, ground_truth_init=1, reg_scale=(0, 1), init_seed=(0, 1, 2)):
        """
        Create and build the PairNet
        :param kernel: Kernel object
        :param n_input: input dimension
        :param reg_scale: l1, and l2 regularization strength
        """

        self.graph = tf.Graph()
        self.kernel = kernel
        self.n_input = n_input
        self.kernel_pre_const = kernel.kernel_pre
        self.kernel_post_const = kernel.kernel_post
        self.kernel_post_post_const = kernel.kernel_post_post
        self.kernel_scale = kernel.kernel_scale
        self.kernel_bias = kernel.bias
        self.reg_scale = reg_scale
        self.ground_truth_init = ground_truth_init
        self.init_seed = init_seed
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
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            kernel_len = self.kernel.len_kernel

            # Apply mask to train only half of the kernel
            mask = np.zeros(shape=[kernel_len, 1])
            mask[:int((kernel_len - 1) / 2) + 1, 0] = 1

            if self.ground_truth_init:  # Not in training mode
                self.kernel_pre = tf.get_variable(shape=self.kernel_pre_const.shape, dtype=tf.float32,
                                                  initializer=tf.constant_initializer(self.kernel_pre_const),
                                                  name='const_pre_kernel')
                self.kernel_pre = tf.multiply(self.kernel_pre, mask)
                self.kernel_post = tf.get_variable(shape=self.kernel_post_const.shape, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(self.kernel_post_const),
                                                   name='const_post_kernel')
                self.kernel_post = tf.multiply(self.kernel_post, mask)
                self.bias = tf.get_variable(shape=self.kernel_bias.shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.kernel_bias), name='bias')
            else:
                self.kernel_pre = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1],
                                                  initializer=self.random_init(self.init_seed[0]), name='pre_kernel')
                self.kernel_pre = tf.multiply(self.kernel_pre, mask)
                self.kernel_post = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1],
                                                   initializer=self.random_init(self.init_seed[1]), name='post_kernel')
                self.kernel_post = tf.multiply(self.kernel_post, mask)

                self.bias = tf.Variable(0, dtype=tf.float32, name='bias')

            self.hp_ker = tf.get_variable(name='hp_ker', shape=[3, 1],
                                          initializer=tf.constant_initializer(np.array([1, -2, 1]).reshape(-1, 1)))

            # Generate pair_term1 and pair_term2
            self.y_pre = self.conv_1d(data=self.x_pre, kernel=self.kernel_pre)
            self.y_post = self.conv_1d(data=self.x_post, kernel=self.kernel_post)

            self.pair_term1 = tf.reduce_sum(tf.multiply(self.y_pre, tf.expand_dims(self.x_post, axis=2)), 1)
            self.pair_term2 = tf.reduce_sum(tf.multiply(self.y_post, tf.expand_dims(self.x_pre, axis=2)), 1)

            self.prediction = self.pair_term1 - self.pair_term2
            self.mse = tf.reduce_mean(tf.square(self.prediction - self.target))

            self.alpha_l1 = self.reg_scale[0]
            self.alpha_l2 = self.reg_scale[1]

            # Enforce that l2 norm of kernel equals to 1, pushing the learning of fc layer
            l2_normalization = (self.l2_loss_unit(self.kernel_pre) + self.l2_loss_unit(self.kernel_post))/2

            # Reduce oscillation of the kernel
            hp_l2_loss = (self.l2_loss(self.kernel_hp(self.kernel_pre))
                          + self.l2_loss(self.kernel_hp(self.kernel_post)))/2

            self.loss = self.mse + self.alpha_l2 * l2_normalization + self.alpha_l1 * hp_l2_loss

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
        # y = tf.squeeze(tf.nn.conv1d(value=x, filters=kernel, stride=1, padding='SAME'))
        y = tf.nn.conv1d(value=x, filters=kernel, stride=1, padding='SAME')

        return y

    def regularization(self):

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.01, scope=None)
        l1_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [self.kernel_pre, self.kernel_post])

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01, scope=None)
        l2_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, [self.kernel_pre, self.kernel_post])

        return self.alpha_l1 * l1_penalty + self.alpha_l2 * l2_penalty

    def random_init(self, seed):
        # Seed the random initializer for kernel_pre and kernel_post
        initializer = tf.contrib.layers.xavier_initializer(seed=seed)

        return initializer
    
    def l2_loss(self, kernel_raw):
        return tf.nn.l2_loss(kernel_raw)
    
    def l2_loss_unit(self, kernel_raw):
        return tf.square(tf.norm(kernel_raw, ord=2) - 1)
    
    def kernel_hp1(self, w):
        """
        Use the laplacian filter to detect the edge
        :param w:
        :return:
        """
        w_out = tf.nn.conv1d(tf.expand_dims(input=w, axis=0), tf.expand_dims(self.hp_ker, axis=2),
                             stride=1, padding='SAME')
        return w_out

    def kernel_hp2(self, w, len=11):
        """
        Use the hamming filter to detect the frequency range that you want to eliminate
        :param w:
        :return:
        """
        window = signal.hamming(len)
        hp_ker2 = tf.constant(window / np.sum(window), shape=(len, 1), dtype=tf.float32)
        w_out = w - tf.nn.conv1d(tf.expand_dims(input=w, axis=0), tf.expand_dims(hp_ker2, axis=2),
                             stride=1, padding='SAME')
        return w_out
    
# ==============================================================================================


class TripNet(PairNet):

    def __init__(self, kernel=None, n_input=None, ground_truth_init=1, reg_scale=(1, 1, 1), len_scale=21, init_seed=(0, 1, 2, 3)):

        self.len_scale = len_scale

        super(TripNet, self).__init__(kernel=kernel, n_input=n_input, ground_truth_init=ground_truth_init,
                                      reg_scale=reg_scale, init_seed=init_seed)
        """
        Create and build the PairNet
        :param kernel: Kernel object
        :param n_input: input dimension
        :param reg_scale: l1, and l2 regularization strength
        """

    def build(self):
        pass
        """
        Build the architecture of the network
        :param: n_input, length of input
        :return:
        """
        with self.graph.as_default():

            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input, 2], name='inputs')
            self.x_pre, self.x_post = tf.unstack(self.inputs, axis=2)
            # Shift the spike trains instead of kernel
            self.x_post_post = tf.concat([self.x_post[:, 1:], tf.expand_dims(self.x_post[:, 0], axis=1)], axis=1)
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            # Use different kernel length for pair and triplet term
            kernel_len = self.kernel.len_kernel
            len_trip = self.kernel.len_trip
            
            # Apply mask to train only half of the kernel 
            mask = np.zeros(shape=[kernel_len, 1])
            mask2 = np.zeros(shape=[kernel_len, 1])
            mask3 = np.zeros(shape=[len_trip, 1])
            mask[:int((kernel_len-1)/2), 0] = 1
            mask2[:int((kernel_len-1)/2)+1, 0] = 1
            mask3[:int((len_trip - 1)/2)+1, 0] = 1
            
            if self.ground_truth_init:  # Not in training mode
                self.kernel_pre = tf.get_variable(shape=self.kernel_pre_const.shape, dtype=tf.float32,
                                                  initializer=tf.constant_initializer(self.kernel_pre_const),
                                                  name='const_pre_kernel')
                self.kernel_pre = tf.multiply(self.kernel_pre, mask2)
                self.kernel_post = tf.get_variable(shape=self.kernel_post_const.shape, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(self.kernel_post_const),
                                                   name='const_post_kernel')
                self.kernel_post = tf.multiply(self.kernel_post, mask2)
                self.kernel_post_post = tf.get_variable(shape=self.kernel_post_post_const.shape, dtype=tf.float32,
                                                        initializer=tf.constant_initializer(self.kernel_post_post_const),
                                                        name='const_post_post_kernel')
                self.kernel_post_post = tf.multiply(self.kernel_post_post, mask3)
                self.fc_w = tf.get_variable(shape=self.kernel_scale.shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.kernel_scale),
                                            name='const_fully_connect_weights')
                self.bias = tf.get_variable(shape=self.kernel_bias.shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.kernel_bias),
                                            name='const_bias')
            else:
                self.kernel_pre = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1],
                                                  initializer=self.random_init(self.init_seed[0]), name='pre_kernel')
                self.kernel_pre = tf.multiply(self.kernel_pre, mask2)
                self.kernel_post = tf.get_variable(dtype=tf.float32, shape=[kernel_len, 1],
                                                   initializer=self.random_init(self.init_seed[1]), name='post_kernel')
                self.kernel_post = tf.multiply(self.kernel_post, mask)
                self.kernel_post_post = tf.get_variable(dtype=tf.float32, shape=[len_trip, 1],
                                                        initializer=self.random_init(self.init_seed[2]), name='post_post_kernel')
                self.kernel_post_post = tf.multiply(self.kernel_post_post, mask3)

                self.fc_w =tf.get_variable(dtype=tf.float32, shape=self.kernel_scale.shape,
                                           initializer=self.random_init(self.init_seed[3]), name='fully_connect_weights')
                self.bias = tf.Variable(0, dtype=tf.float32, name='bias')

            self.hp_ker = tf.get_variable(name='hp_ker', shape=[3, 1],
                                          initializer=tf.constant_initializer(np.array([1, -2, 1]).reshape(-1, 1)))
            
            self.y_pre = self.conv_1d(data=self.x_pre, kernel=self.kernel_pre)
            self.y_post = self.conv_1d(data=self.x_post, kernel=self.kernel_post)
            self.y_post_post = self.conv_1d(data=self.x_post, kernel=self.kernel_post_post)

            self.pair_term1 = tf.reduce_sum(tf.multiply(self.y_pre, tf.expand_dims(self.x_post, axis=2)), 1)
            self.pair_term2 = tf.reduce_sum(tf.multiply(self.y_post, tf.expand_dims(self.x_pre, axis=2)), 1)
            self.trip_term = tf.reduce_sum(tf.multiply(tf.multiply(self.y_pre, self.y_post_post),
                                                       tf.expand_dims(self.x_post_post, axis=2)), 1)

            self.terms = tf.concat([self.pair_term1, -1 * self.pair_term2, self.trip_term], axis=1)

            self.prediction = tf.matmul(self.terms, tf.expand_dims(self.fc_w, axis=1)) + self.bias

            self.mse = tf.reduce_mean(tf.square(self.prediction - self.target))

            # Enforce l2 norm of each kernel to be 1, promote the unit learning of fc layer.
            l2_normalization = (self.l2_loss_unit(self.kernel_pre)+self.l2_loss_unit(self.kernel_post) +
                                self.l2_loss_unit(self.kernel_post_post))/3

            # Use edge detection filter to remove high frequency oscillation
            hp_pre = self.kernel_hp1(self.kernel_pre)
            hp_post = self.kernel_hp1(self.kernel_post)
            hp_post_post = self.kernel_hp1(self.kernel_post_post)
            hp_loss1 = (self.l2_loss(hp_pre)+self.l2_loss(hp_post)+self.l2_loss(hp_post_post)) / 3

            # Use Hamming window to promote smoothness
            hp_pre2 = self.kernel_hp2(self.kernel_pre[:int((kernel_len-1)/2)-2], len=self.len_scale)
            hp_post2 = self.kernel_hp2(self.kernel_post[:int((kernel_len-1)/2)-1], len=self.len_scale)
            hp_post_post2 = self.kernel_hp2(self.kernel_post_post, len=self.len_scale)
            hp_loss2 = (self.l2_loss(hp_pre2) + self.l2_loss(hp_post2) + self.l2_loss(hp_post_post2)) / 3

            self.alpha_l1 = self.reg_scale[0]
            self.alpha_l2 = self.reg_scale[1]
            self.alpha_l3 = self.reg_scale[2]

            self.loss = self.mse + self.alpha_l1 * l2_normalization + self.alpha_l2 * hp_loss1 + self.alpha_l3 * hp_loss2
