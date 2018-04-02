"""
    Perform hyperparameter search for hippocampal dataset with 3 differen type of data augmentation. Save the network and best parameters to specific folder in /model"
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelval.kernel import KernelGen
from modelval.trainer import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from modelval import pairptl, network, trainer, dataset
from modelval.ArbDataGen import arb_w_gen
import datajoint as dj
import tempfile
from shutil import copyfile
from os.path import join
import pdb

schema = dj.schema('shuang_hyperpara_hippo_schema', locals())


@schema
class Network(dj.Lookup):
    definition = """
    # Type of network to validate
    network_id          : smallint # l1 regularization id
    ---
    network_name       : varchar(100) # name of the network
    """

    @property
    def contents(self):
        return [(1, 'triplet')]


@schema
class Dataset(dj.Lookup):
    definition = """
    # Type of spike train data to use for performing model validation
    dataset_id          : smallint # l1 regularization id
    ---
    dataset_name      : varchar(100) # name of the dataset
    """

    @property
    def contents(self):
        return [(1, 'Hippocampus')]

@schema
class Dataaug(dj.Lookup):
    definition = """
    # Data augmentation strategy
    dataaug_id          : smallint # data augmentation strategy name
    ---
    dataaug_name      : varchar(100) # data augmentation strategy id
    """

    @property
    def contents(self):
        return [(0, 'Raw data'), (1, 'GP_mean'), (2, 'GP_sample')]

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
        return [(0, 0.0)]


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
        return [(0, 1.0)]

@schema
class SmoothLp(dj.Lookup):
    definition = """
    # Smoothness filter with Laplacian filter
    # Doesn't apply for model validation
    sm_lp_id          : smallint # laplacian filter smoothing id
    ---
    alpha_sm_lp       : float # regularization constant
    """

    @property
    def contents(self):
        return [(0, 1.0)]

@schema
class SmoothHam(dj.Lookup):
    definition = """
    # Smoothness filter with hamming window
    # Doesn't apply for model validation
    sm_hm_id          : smallint # Hamming window smoothness id
    ---
    alpha_sm_hm       : float # regularization constant
    """

    @property
    def contents(self):
        return [(0, 1.0)]

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
    contents = [(1, 0), (1, 100)]


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
    scale             : longblob       # weight for output layer
    """

    def _make_tuples(self, key):
        print('Populating', key, flush=True)

        # Fetch parameter from the server and save temporally locally, and generate spike trains and targets
        if self.tmppath is None:
            self.tmppath = tempfile.mkdtemp()

        data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')


        # Generate data
        ptl_list = [1, 2, 3, 4]
        data_select = data[data['ptl_idx'].isin(ptl_list)]








        # Insert values for STDP
        dt = np.arange(-100, 100, 2)
        for i in range(len(dt)):
            new_try1 = data[data['ptl_idx'] == 1].iloc[0]
            new_try1['dt1'] = dt[i]
            data_select = data_select.append(new_try1, ignore_index=True)

        # Insert values for Quadruplet protocol
        for i in range(len(dt)):
            if np.abs(dt[i]) > 10:
                new_try2 = data[data['ptl_idx'] == 3].iloc[0]
                new_try2['dt2'] = dt[i]
                data_select = data_select.append(new_try2, ignore_index=True)

        spk_len = int(data[data['ptl_idx'].isin(ptl_list)]['train_len'].max() * 1000 / ker_test.reso_kernel)
        spk_pairs, targets = arb_w_gen(df=data_select, ptl_list=ptl_list, spk_len=spk_len, kernel=ker_test,
                                       aug_times=[10, 10, 10, 10],
                                       net_type=net_name)

        # Build the network
        ground_truth_init = 0

        alpha1 = (L1RegConstant() & key).fetch1['alpha_l1']
        alpha2 = (L2RegConstant() & key).fetch1['alpha_l2']
        reg_scale = (alpha1, alpha2)

        if net_name == 'triplet':
            toy_data_net = network.TripNet(kernel=ker_test, ground_truth_init=ground_truth_init, reg_scale=reg_scale,
                                           n_input=spk_pairs.shape[1])
        elif net_name == 'pair':
            toy_data_net = network.PairNet(kernel=ker_test, ground_truth_init=ground_truth_init, reg_scale=reg_scale,
                                           n_input=spk_pairs.shape[1])
        else:
            print('Wrong network!!!')
            return

        # Create the trainer
        save_dir = '/'.join(('/src/Plasticity_Ker/model', data_name, net_name))
        toy_net_trainer = Trainer(toy_data_net.mse, toy_data_net.loss, input_name=toy_data_net.inputs,
                                  target_name=toy_data_net.target, save_dir=save_dir,
                                  optimizer_config={'learning_rate': toy_data_net.lr})

        # Obtain the training and validation data

        X_train, X_vali, y_train, y_vali = train_test_split(spk_pairs, targets, test_size=0.1)
        train_data = dataset.Dataset(X_train, y_train)
        vali_data = dataset.Dataset(X_vali, y_vali)

        # Learn the kernel from random initialization
        learning_rate = (LearningRate() & key).fetch1['learning_rate']
        iterations = (Loops() & key).fetch1['iterations']
        min_error = 1e-14
        mse_temp = 1
        max_iteration = 20
        iter_count = 0
        while (mse_temp > min_error) & (iter_count < max_iteration):
            toy_net_trainer.train(train_data, vali_data, batch_size=128, min_error=min_error,
                                  feed_dict={toy_data_net.lr: learning_rate})
            mse_temp = toy_net_trainer.evaluate(ops=toy_data_net.mse, inputs=X_vali, targets=y_vali,
                                                feed_dict={toy_data_net.lr: learning_rate})
            learning_rate = learning_rate / 3
            iter_count += 1

        kernel_pre = toy_net_trainer.evaluate(ops=toy_data_net.kernel_pre)
        kernel_post = toy_net_trainer.evaluate(ops=toy_data_net.kernel_post)
        fc_w = toy_net_trainer.evaluate(ops=toy_data_net.fc_w)

        if net_name == 'triplet':
            kernel_post_post = toy_net_trainer.evaluate(ops=toy_data_net.kernel_post_post)
        elif net_name == 'pair':
            kernel_post_post = np.zeros((1, 1))
        else:
            print('Wrong network!!!')
            return

        mse = toy_net_trainer.evaluate(ops=toy_data_net.mse, inputs=X_vali, targets=y_vali,
                                       feed_dict={toy_data_net.lr: learning_rate})

        cost = toy_net_trainer.evaluate(ops=toy_data_net.loss, inputs=X_vali, targets=y_vali,
                                        feed_dict={toy_data_net.lr: learning_rate})

        key['val_error'] = mse
        key['val_loss'] = cost
        key['scale'] = fc_w
        key['pre_kernel'] = kernel_pre
        key['post_kernel'] = kernel_post
        key['post_post_kernel'] = kernel_post_post
        self.insert1(key)
