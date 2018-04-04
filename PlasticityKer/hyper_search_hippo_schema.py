"""
    Perform hyperparameter search for hippocampal dataset with 3 differen type of data augmentation. Save the network and best parameters to specific folder in /model"
"""

import numpy as np
from sklearn.model_selection import train_test_split
from modelval.trainer import Trainer
from modelval import pairptl, network, trainer, dataset
import datajoint as dj
from modelval.model_fit_cv import ModelFitCV
import tempfile

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
        return [(1, 'hippocampus')]

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
        return [(0, 'Raw data'), (1, 'gp_mean'), (2, 'gp_sample')]

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

        # Load the data

        data_type = (Dataset() & key).fetch1['dataset_name']
        data_aug = (Dataaug() & key).fetch1['dataaug_name']
        networ_type = (Network() & key).fetch1['network_name']
        alpha1 = (L1RegConstant() & key).fetch1['alpha_l1']
        alpha2 = (L2RegConstant() & key).fetch1['alpha_l2']
        alpha_lp = (SmoothLp() & key).fetch1['alpha_sm_lp']
        alpha_hm = (SmoothHam() & key).fetch1['alpha_sm_hm']
        seed = (Seed() & key).fetch1['seed']

        save_dir = '/'.join(('/src/Plasticity_Ker/model/', data_type, data_aug, networ_type,
                             str(alpha1), str(alpha2), str(alpha_lp), str(alpha_hm), str(seed)))

        vali_err, w_pre, w_post, w_post_post, fc_w, bias, scale = ModelFitCV(data_type=data_type, data_aug=data_aug, test_fold_num=0,
                              vali_split=5, save_dir_set=save_dir, random_seed=seed)

        key['val_error'] = vali_err
        key['fc_w'] = fc_w
        key['pre_kernel'] = w_pre
        key['post_kernel'] = w_post
        key['post_post_kernel'] = w_post_post
        key['bias'] = bias
        key['scale'] = scale
        self.insert1(key)
