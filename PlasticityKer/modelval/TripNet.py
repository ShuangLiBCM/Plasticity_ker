# Realize triplet-based network

import numpy as np
import tensorflow as tf


def target_gen(data_train_pre=None, data_train_post=None, data_train_post_shift=None, kernel_pre_post=None, kernel_post_pre=None, kernel_post_post=None, len_ker=51, bias=0):

        # Network Parameters
        n_labels = data_train_pre.shape[0]  # label dimension: ~500
        data_len = data_train_pre.shape[1]
        
        g = tf.Graph()

        with g.as_default():
            # tf Graph input
            x_pre = tf.placeholder(tf.float32, [None, data_len])
            x_post = tf.placeholder(tf.float32, [None, data_len])
            
            # Initialize with the original kernel
            init_pre_post = tf.constant(kernel_pre_post, dtype=tf.float32, shape=[len_ker, 1, 1])
            init_post_pre = tf.constant(kernel_post_pre, dtype=tf.float32, shape=[len_ker, 1, 1])
            init_post_post = tf.constant(kernel_post_post, dtype=tf.float32, shape=[len_ker, 1, 1])
            # Store layers weight & bias
            weights = {
                'wc_pre_post': tf.get_variable("wc_pre_post", initializer=init_pre_post),
                'wc_post_pre': tf.get_variable("wc_post_pre", initializer=init_post_pre),
                'wc_post_post': tf.get_variable("wc_post_post", initializer=init_post_post),
            }

            # Construct model
            pred= conv_net(x_pre, x_post,n_labels, data_len, weights, bias)

            # Initializing the variables
            init = tf.global_variables_initializer()

            #  Launch the gragh
            sess = tf.Session(graph=g)
            sess.run(init)
            output= sess.run([pred], feed_dict={x_pre: data_train_pre, x_post: data_train_post})

        return output
# Create some wrappers for simplicity
def conv1d(x, W):
    # Conv1D wrapper, without bias or relu activation
    y = tf.nn.conv1d(x, W, stride=1, padding='SAME')
    return y

# Create model
def conv_net(x_pre, x_post, n_labels, data_len, weights, bias):
    # Reshape input picture            
    x_pre = tf.reshape(x_pre, shape=[-1, data_len, 1])
    x_post = tf.reshape(x_post, shape=[-1, data_len, 1])
        
    # Convolution Layer
    y_pre_post = tf.nn.conv1d(x_pre, weights['wc_pre_post'], stride=1, padding='SAME')
    y_post_pre = tf.nn.conv1d(x_post, weights['wc_post_pre'], stride=1, padding='SAME')
    y_post_post = tf.nn.conv1d(x_post, weights['wc_post_post'], stride=1, padding='SAME')
    y_pre_post_post = tf.multiply(y_pre_post, y_post_post)
    
    # Implement the shift 
    fc1_tmp = tf.multiply(y_pre_post, x_post)
    fc2_tmp = tf.multiply(y_post_pre,  x_pre)
    fc3_tmp = tf.multiply(y_pre_post_post, x_post)
    # Fully connected layer
    fc1 = tf.reduce_sum(fc1_tmp, 1)
    fc2 = tf.reduce_sum(fc2_tmp, 1)
    fc3 = tf.reduce_sum(fc3_tmp, 1)
    # Output, class prediction
    output = fc1-fc2+fc3+bias
    # output = fc1
    return output

# Generate kernel based on Gerstner's parameter
# Generate kernel with designated shape
# define 3 exponential decay kernel with different time constant
def ker_gen_shift(a, tau, reso_set=1, len_ker=401):
    tau_pre_post = tau[0]/reso_set  # ms
    tau_post_pre = tau[2]/reso_set # ms
    tau_post_post = tau[3]/reso_set # ms
    
    half_pt = int((len_ker - 1)/2+1)
    half_shift = half_pt - 1
    
    x1 = np.linspace(0, half_pt, half_pt)
    x2 = np.linspace(0, half_shift, half_shift)
    
    ker_pre_post = np.zeros([len_ker,1])
    ker_post_pre = np.zeros([len_ker,1])
    ker_post_post = np.zeros([len_ker,1])

    # ker_pre_post[25] = 1
    # ker_post_pre[25] = 1
    # ker_post_post[25] = 1

    ker_pre_post[:half_pt] =  a[0] * exp_ker(x1, tau_pre_post, half_pt).reshape([half_pt,1])
    ker_post_pre[:half_pt] = a[2] * exp_ker(x1, tau_post_pre, half_pt).reshape([half_pt,1])
    if a[0] > 1e-10:
        ker_post_post[:half_shift] = a[3]/a[0] * exp_ker(x2, tau_post_post,half_shift).reshape([half_shift,1])/ np.exp(-1/tau_pre_post)
    else:
        ker_post_post[:half_shift] = a[3] * exp_ker(x2, tau_post_post, half_shift).reshape([half_shift,1])
    return ker_pre_post, ker_post_pre, ker_post_post, len_ker

def ker_gen(a, tau, reso_set=1, len_ker=401):
    tau_pre_post = tau[0]/reso_set  # ms
    tau_post_pre = tau[2]/reso_set # ms
    tau_post_post = tau[3]/reso_set # ms
    
    half_pt = int((len_ker - 1)/2+1)
    
    x1 = np.linspace(0, half_pt, half_pt)
    
    ker_pre_post = np.zeros([len_ker,1])
    ker_post_pre = np.zeros([len_ker,1])
    ker_post_post = np.zeros([len_ker,1])

    # ker_pre_post[25] = 1
    # ker_post_pre[25] = 1
    # ker_post_post[25] = 1

    ker_pre_post[:half_pt] =  a[0] * exp_ker(x1, tau_pre_post, half_pt).reshape([half_pt,1])
    ker_post_pre[:half_pt] = a[2] * exp_ker(x1, tau_post_pre, half_pt).reshape([half_pt,1])
    if a[0] > 1e-10:
        ker_post_post[:half_pt] = a[3]/a[0] * exp_ker(x1, tau_post_post,half_pt).reshape([half_pt,1])
    else:
        ker_post_post[:half_pt] = a[3] * exp_ker(x1, tau_post_post, half_pt).reshape([half_pt,1])
    return ker_pre_post, ker_post_pre, ker_post_post, len_ker

def exp_ker(x, tau,length):
    z = np.exp(-(length - x)/tau)
    return z