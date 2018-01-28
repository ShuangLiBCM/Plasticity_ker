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
    l1_id          : smallint # l1 regularization id
    ---
    alpha_l1       : float # regularization constant
    """

    @property
    def contents(self):
        return list(enumerate(np.hstack(([0], 10 ** np.arange(-1., 0)))))

@schema
class Dataset(dj.Lookup):
    definition = """
    # L2 regularization constants for kernel model selection
    l2_id          : smallint # l1 regularization id
    ---
    alpha_l2       : float # regularization constant
    """

    @property
    def contents(self):
        return list(enumerate(np.hstack(([0], 10 ** np.arange(-4., -3.)))))


# Generate kernel

# Load data

# Generate data of specific protocol (Train, targets)

# Build the network

# Create the trainer

# Train the model

# Save the model and best parameter

