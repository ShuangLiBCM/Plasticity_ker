"""
Implement non-stationary gaussian regression
"""
import numpy as np

def gp_regessor(x, y, x_test, sigma_obs, sigma_kernel, bias=1, if_stationary=True):
    """
    Perform non-stationary Gaussian Regression on training data and generate prediction for testing data
    :param x: x axis of training data, m data points with d dimension each (m, d)
    :param y: y axis of training data, (m, 1)
    :param x_test: x axis of testing data, k data points with d dimension each (k,d)
    :param sigma_obs:
    :param sigma_kernel:
    :param if_nonstationary: whether to enable the nonstationary component
    :return:
    """
    y_bias = np.squeeze(np.mean(y, axis=0))    # Assume zeros mean, remove bias first, add at the back
    
    K = cov_kernel(x, x, sigma_kernel, bias, if_stationary)
    K_test = cov_kernel(x_test, x, sigma_kernel,bias, if_stationary)

    L = np.linalg.cholesky(K + np.square(sigma_obs) * np.eye(K.shape[0]))
    alpha = np.linalg.lstsq(L.transpose(), np.linalg.lstsq(L, y-y_bias)[0])[0]
    f = np.dot(K_test, alpha)

    f = f + y_bias
    v = np.linalg.lstsq(L, K_test.transpose())[0]
    v_f = cov_kernel(x_test, x_test, sigma_kernel, bias, if_stationary)- np.dot(v.transpose(), v)

    lp = -1 / 2 * (y - y_bias).transpose() * alpha - np.sum(np.log(K)) - x.shape[0]/2 * np.log(2 * np.pi)

    return f, v_f, lp

def cov_kernel(x1, x2, prec, bias, if_stationary=True):

    K = np.zeros((x1.shape[0], x2.shape[0]))

    for i in range(x1.shape[0]):
        dist = x1[i, :] - x2
        
        if not if_stationary:
            len_scale = np.abs(x1[i, :])
        else:
            len_scale = 1
        K[i, :] = np.exp(-0.5 * np.square(dist/(prec*np.sqrt(len_scale)+3))).transpose() # Sum over the dimension d

    return K
