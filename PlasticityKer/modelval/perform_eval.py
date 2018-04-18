"""
    Evaluate performance and making figures
"""
import numpy as np

def R2_corr(predictions, targets, x_fit):

    R2 = 1 - np.sum(np.square(predictions - targets)) / np.sum(np.square(targets - np.mean(targets)))
    # Calculate correlation coefficient
    corr = np.sum(np.dot((predictions - np.mean(predictions)).T, targets - np.mean(targets))) / np.std(
        predictions) / np.std(targets) / len(targets)
    b1 = np.sum(np.dot((predictions - np.mean(predictions)).T, targets - np.mean(targets))) / np.sum(
        np.square(targets - np.mean(targets)))
    b2 = np.mean(predictions) - np.mean(targets) * b1
    y_fit = x_fit * b1 + b2
    return R2, corr, y_fit

def pred_model_stdp(x, reso, pred_model):
    """
    Given the prediction of model for stdp and interested time x, return the error with length x
    :param x: array of time for points interested in generating prediction
    :param reso: sampling resolution
    :param pred_model: fitted response
    :return:
    """

    return

def pred_net_stdp(x, reso, pred_net):
    """
        Given the prediction of model for stdp and interested time x, return the error with length x
        :param x: array of time for points interested in generating prediction
        :param reso: sampling resolution
        :param pred_model: fitted response
        :return:
    """
    return
