"""
    Perform model validation for two network (PairNet, TripNet) and 3 datasets (STDP alone, Hippocampal data,
    Visual cortex data). Save the network and best parameters to specific folder in /model"
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

schema = dj.schema('shuang_plasticity_schema')

@schema
class Protocol(dj.Manual):
    definition = """
    # Load in pairing protocol paramter
    ptl_index:           int  auto_increment  # Index for each protocol
    ---
    ptl_idx:             smallint   # Type of protocol
    pre_spk_num:         smallint   # Number of presynatpic spike
    pre_spk_freq:        smallint   # Frequency of presynaptic spike
    post_spk_num:        smallint   # Number of postsynaptic spike
    post_spk_freq:       smallint   # Frequency of postsynaptic spike
    ptl_occ:             smallint   # Number of protocol repitition
    ptl_freq:            smallint   # Frequency of protocol
    dt1:                 float      # pre-post time difference in ms
    dt2:                 float      # pre-pre time difference in ms
    dt3:                 float      # post-post time difference in ms
    dw_mean:             float      # change in w or mean change in w
    dw_ste:              float      # std change in w
    train_len:           smallint   # length of spike train
    """


@schema
class Dataset(dj.Lookup):
    definition = """
    # Type of spike train data to use for performing model validation
    dataset_id          : smallint # id for type of dataset
    ---
    dataset_name        : varchar(100) # name of the dataset
    """

    @property
    def contents(self):
        return [(1, 'hippocampus'), (2, 'visual cortex')]


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
        return [(0, 'raw_data'), (1, 'gp_mean'), (2, 'gp_samp')]


@schema
class Data_seed(dj.Lookup):
    definition = """
    # Seed for random initialization of the network
    random_start              : tinyint # 1 if random start, 0 if start from ground truth, seed is ignored when 0
    seed                      : int # random seed
    ---
    """
    contents = [(1, 0), (1, 100)]


@schema
class Kernel(dj.Manual):
    definition = """
    # Seed for random initialization of the network
    kernel_id         :  int   # id for kernel
    ---
    a_fast        : float    # a fast
    a_slow       : float    # a_slow
    a_fast_      : float    # a_fast2
    a_slow_         : float       # a_slow2
    tau_fast          : float       # tau_fast
    tau_slow             : float       # tau_slow
    tau_fast_        : float       # tau_fast_2
    tau_slow_             : float       # tau_slow2
    """


@schema
class Spikes_dw(dj.Computed):
    definition = """
    -> Protocol
    -> Dataset
    -> Dataaug
    -> Data_seed
    -> Kernel
    ---
    spikes:            longblob
    weights:           float
    """

    def make(self, key):
        pass


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
        return [(0, 'pair'), (1, 'triplet')]


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
class ModelSeed(dj.Lookup):
    definition = """
    # Seed for random initialization of the network
    random_start              : tinyint # 1 if random start, 0 if start from ground truth, seed is ignored when 0
    ---
    seed                      : int # random seed
    """
    contents = [(0, 0)]


@schema
class ModelVal(dj.Computed):
    definition = """
    # model selection for learning rule learner
    -> Network
    -> Spikes_dw
    -> L1RegConstant
    -> L2RegConstant
    -> Loops
    -> LearningRate
    -> ModelSeed
    ---
    pre_kernel        : longblob    # presynaptic kernel
    post_kernel       : longblob    # postsynaptic kernel
    post_post_kernel  : longblob    # postsynaptic kernel for higher order interaction
    val_error         : float       # mse on validation set
    val_loss          : float       # total loss on validation set
    scale             : longblob       # weight for output layer
    """

    def make(self, key):
        pass


@schema
class Modelselection(dj.Computed):
    definition = """
    # model selection for learning rule learner
    -> Network
    -> Spikes_dw
    -> L1RegConstant
    -> L2RegConstant
    -> Loops
    -> LearningRate
    -> ModelSeed
    ---
    pre_kernel        : longblob    # presynaptic kernel
    post_kernel       : longblob    # postsynaptic kernel
    post_post_kernel  : longblob    # postsynaptic kernel for higher order interaction
    val_error         : longblob    # mse on validation set
    bias              : longblob    # bias of the output layer
    scale             : longblob    # weight for output layer
    fc_w              : longblob    # fully connected weights
    """

    def make(self, key):
        pass