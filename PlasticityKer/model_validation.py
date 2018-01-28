"""
    Perform model validation for two network (PairNet, TripNet) and 3 datasets (STDP alone, Hippocampal data,
    Visual cortex data). Save the network and best parameters to specific folder in /model"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from modelval import pairptl, network, trainer, dataset
from modelval.ArbDataGen import arb_w_gen
import datajoint as dj
import tempfile
from shutil import copyfile
from os.path import join

schema = dj.schema('shuang_model_validation_schema', locals())


@schema
class Network(dj.Lookup):
    definition = """
    # Type of network to validate
    network_id          : smallint # l1 regularization id
    ---
    network_name       : string # name of the network
    """

    @property
    def contents(self):
        return ['PairNet', 'TripNet']

@schema
class Dataset(dj.Lookup):
    definition = """
    # Type of spike train data to use for performing model validation
    dataset_id          : smallint # l1 regularization id
    ---
    dataset_name      : string # name of the dataset
    """

    @property
    def contents(self):
        return ['STDP', 'Hippocampus', 'VisualCortex']

@schema
class L1RegConstant(dj.Lookup):
    definition = """
    # L1 regularization constants for kernel model selection
    # Doesn't apply for model validation
    l1_id          : smallint # l1 regularization id
    ---
    alpha_l1       : float # regularization constant
    """

    @property
    def contents(self):
        return [0]


@schema
class L2RegConstant(dj.Lookup):
    definition = """
    # L2 regularization constants for kernel model selection
    # Doesn't apply for model validation
    l2_id          : smallint # l1 regularization id
    ---
    alpha_l2       : float # regularization constant
    """

    @property
    def contents(self):
        return [0]

# @schema
# class InitBias(dj.Lookup):
#     definition = """
#     # whether to initialize with constant bias or random bias
#     # Doesn't apply for model validation
#     random_bias               : tinyint # 1 if with random bias initialization, 0 if start with constant
#     constant_seed              : int # random or constant intialization
#     ---
#     """
#     contents = [(0, 0), (0, 2), (1, 2018)]

@schema
class Loops(dj.Lookup):
    definition = """
    # number of iterations
    iterations      : int   # number of iterations
    ---
    """

    contents = [(5,)]

@schema
class LearningRate(dj.Lookup):
    definition = """
    # learning rate
    lr_id   : tinyint
    ---
    learning_rate      : float # initial learning_rate for using the Adam optimizer
    """

    contents = [(1, 0.001)]


@schema
class Seed(dj.Lookup):
    definition = """
    # Seed for random initialization of the network
    random_start              : tinyint # 1 if random start, 0 if start from ground truth, seed is ignored when 0
    seed                      : int # random seed
    ---
    """
    contents = [(0, 0), (1, 723), (1, 607)]

@schema
class ModelSelection(dj.Computed):
    tmppath = None

    definition = """
    # model selection for learning rule learner
    -> Network
    -> Dataset
    -> L1RegConstant
    -> L2RegConstant
    -> Loops
    -> LearningRate
    -> Seed
    ---
    pre_kernel        : longblob    # presynaptic kernel
    post_kernel       : longblob    # postsynaptic kernel
    post_post_kernel  : longblob    # postsynaptic kernel for higher order interaction
    val_error         : float       # mse on validation set
    val_loss          : float       # total loss on validation set
    """

    def _make_tuples(self, key):
        print('Populating', key, flush=True)

        # train one model with values from key here
        # if key['random_start']:
        # Data generation
        # data = kernel_data_aug.DataAug(jitter = key['jitters'])
        # data_input = data.data_input
        # data.data_gen()

        # Fetch data from the server and save temporally locally
        if self.tmppath is None:
            self.tmppath = tempfile.mkdtemp()
            copyfile('/data/data_input_1k.npy', join(self.tmppath, 'data_input_1k.npy'))
            copyfile('/data/target_final_1k.npy', join(self.tmppath, 'target_final_1k.npy'))
            copyfile('/data/pre_kernel_1k.npy', join(self.tmppath, 'pre_kernel_1k.npy'))
            copyfile('/data/post_kernel_1k.npy', join(self.tmppath, 'post_kernel_1k.npy'))

        data_input = np.load(join(self.tmppath, 'data_input_1k.npy'))
        target_final = np.load(join(self.tmppath, 'target_final_1k.npy'))
        kernel_pre = np.load(join(self.tmppath, 'pre_kernel_1k.npy'))
        kernel_post = np.load(join(self.tmppath, 'post_kernel_1k.npy'))
        # Build the model

        dataset = Dataset(inputs=data_input, targets=target_final)
        train_inputs, train_targets = dataset.train_set()
        # validation_inputs, validation_targets = dataset.train_set()

        net = KernelNet(n_input=train_inputs.shape[1], kernel_pre=kernel_pre,
                        kernel_post=kernel_post, len_k=51, if_1k=True,
                        if_rand_ini=key['random_start'], if_rand_bias=key['random_bias'],
                        rand_seed=[key['seed'], key['seed'] + 1, key['constant_seed']])

        def ping(trainer, step, global_step):
            """
            Function to perioidically ping the database to keep the connection alive.
            :param step: current step. Pings every 10,000 steps
            """
            if step % 10000 == 0:
                dj.conn().is_connected

        trainer = Trainer(net.cost, net.input_, net.target_, net.is_training_,
                          optimizer_config={'learning_rate': net.l_rate_}, log_dir=tempfile.mkdtemp())

        # Learning rate
        lr = (LearningRate() & key).fetch1['learning_rate']

        # Strength of L1 normalization
        alpha1 = (L1RegConstant() & key).fetch1['alpha_l1']
        alpha2 = (L2RegConstant() & key).fetch1['alpha_l2']

        trainer.min_total_loss = trainer.evaluate(dataset.validation_inputs, dataset.validation_targets,
                                                  feed_dict={net.l_rate_: lr, net.alpha1_: alpha1, net.alpha2_: alpha2})

        trainer.save_best()
        # Epochs of training
        for i in range(key['iterations']):
            trainer.train(dataset, batch_size=128, max_steps=-1, early_stopping_steps=20, save_freq=-1,
                          feed_dict={net.l_rate_: lr, net.alpha1_: alpha1, net.alpha2_: alpha2}, test_freq=0,
                          callback_fn=ping)
            lr /= 3

        wc_pre = np.hstack(trainer.evaluate(ops=net.wc_pre))
        wc_post = np.hstack(trainer.evaluate(ops=net.wc_post))
        bias = trainer.evaluate(ops=net.bias)

        mse = trainer.evaluate(dataset.validation_inputs, dataset.validation_targets, ops=net.mse,
                               feed_dict={net.l_rate_: lr, net.alpha1_: alpha1, net.alpha2_: alpha2})

        cost = trainer.evaluate(dataset.validation_inputs, dataset.validation_targets, ops=net.cost,
                                feed_dict={net.l_rate_: lr, net.alpha1_: alpha1, net.alpha2_: alpha2})

        key['val_error'] = mse
        key['val_loss'] = cost
        key['pre_kernel'] = wc_pre[0]
        key['post_kernel'] = wc_post[0]
        key['bias'] = bias[0]
        self.insert1(key)
# Generate kernel

# Load data

# Generate data of specific protocol (Train, targets)

# Build the network

# Create the trainer

# Train the model

# Save the model and best parameter

