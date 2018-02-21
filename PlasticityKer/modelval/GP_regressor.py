"""
Implement gaussian regression with non-stationary kernel and noise
"""
import numpy as np

def gp_regessor(x, y, x_test, sigma_obs=1, sigma_kernel=1, bias=1, noise_const=100, scale=5, if_stat_kernel=True, if_stat_noise=True):
    """
    Perform Gaussian Regression on training data and generate prediction for testing data 
    :param x: x axis of training data, m data points with d dimension each (m, d)
    :param y: y axis of training data, (m, 1)
    :param x_test: x axis of testing data, k data points with d dimension each (k,d)
    :param sigma_obs:
    :param sigma_kernel:
    :param if_stat_kernel: if the fitting use stationary prior covariance matrix, default is yes
    :param if_stat_noise: if the fitting use stationary prior noise, default is yes
    :return:
    f: mean of the posterior distribution over function
    v_f: covariance of the posterior distribution over the function
    lp: log likelihood of posterior distribution of function over training data

    Refer to book "Gaussian Processes for Machine Learning" algorithm 2.3

    """
    y_bias = np.squeeze(np.mean(y, axis=0))    # Assume zeros mean, remove bias first, add it back later
    
    K = cov_kernel(x, x, sigma_kernel, bias, scale, if_stat_kernel)
    K_test = cov_kernel(x_test, x, sigma_kernel,bias, scale, if_stat_kernel)
    
    if not if_stat_noise:
        # Implement non-stationary noise that dependents on absolute value of x
        sigma_noise = sigma_obs*np.exp(-1 * np.abs(x)/noise_const)
    else:
        sigma_noise = sigma_obs
        
    L = np.linalg.cholesky(K + np.multiply(np.square(sigma_noise), np.eye(K.shape[0])))
    alpha = np.linalg.lstsq(L.transpose(), np.linalg.lstsq(L, y-y_bias)[0])[0]
    f = np.dot(K_test, alpha)

    f = f + y_bias
    v = np.linalg.lstsq(L, K_test.transpose())[0]
    v_f = cov_kernel(x_test, x_test, sigma_kernel, bias, scale, if_stat_kernel)- np.dot(v.transpose(), v)

    lp = -1 / 2 * (y - y_bias).transpose() * alpha - np.sum(np.log(K)) - x.shape[0]/2 * np.log(2 * np.pi)

    return f, v_f, lp

def cov_kernel(x1, x2, prec, bias, scale, if_stat_kernel=True):

    K = np.zeros((x1.shape[0], x2.shape[0]))

    for i in range(x1.shape[0]):
        dist = x1[i, :] - x2
        
        if not if_stat_kernel:
            len_scale = np.abs(x1[i, :])
        else:
            len_scale = 1
        # Sum over the dimension d
        K[i, :] = scale * np.exp(-0.5 * np.square(dist/(prec*np.sqrt(len_scale)+bias))).transpose()

    return K