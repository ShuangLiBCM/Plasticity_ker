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
    # whether to initialize from ground truth or not
    random_start              : tinyint # 1 if random start, 0 if start from ground truth, seed is ignored when 0
    seed                      : int # random seed
    ---
    """
    contents = [(0, 0), (1, 723), (1, 607)]


# Generate kernel

# Load data

# Generate data of specific protocol (Train, targets)

# Build the network

# Create the trainer

# Train the model

# Save the model and best parameter

