"""
Implement the Gaussian process with non-stationary kernel and noise
"""
import numpy as np

class GP_regressor(object):

    def __init__(self, x, y, x_test, sigma_obs=1, sigma_kernel=1, bias=1, noise_const=100, scale=5, if_stat_kernel=True, if_stat_noise=True):
        """
        Initialize regressor with parameters
        :param x: x axis of training data, m data points with d dimension each (m, d)
        :param y: y axis of training data, (m, 1)
        :param x_test: x axis of testing data, k data points with d dimension each (k,d)
        :param sigma_obs: standard deviation level for noise
        :param sigma_kernel: standard deviation for covariance matrix of the prior
        :param if_stat_kernel: if the fitting use stationary prior covariance matrix, default is yes
        :param if_stat_noise: if the fitting use stationary prior noise, default is yes

        Refer to book "Gaussian Processes for Machine Learning" algorithm 2.3

        """
        self.x = x
        self.y = y
        self.x_test = x_test
        self.sigma_obs = sigma_obs
        self.sigma_kernel = sigma_kernel
        self.bias = bias
        self.noise_const = noise_const
        self.scale = scale
        self.if_stat_kernel = if_stat_kernel
        self.if_stat_noise = if_stat_noise

    def fit(self):
        """
        Perform Gaussian Regression on training data and generate prediction for testing data
        :return:
        f: mean of the posterior distribution over function
        v_f: covariance of the posterior distribution over the function
        lp: log likelihood of posterior distribution of function over training data


        Refer to book "Gaussian Processes for Machine Learning" algorithm 2.3

        """

        y_bias = np.squeeze(np.mean(self.y, axis=0))  # Assume zeros mean, remove bias first, add it back later

        K = self.cov_kernel(self.x, self.x)
        K_test = self.cov_kernel(self.x_test, self.x)

        if not self.if_stat_noise:
            # Implement non-stationary noise that dependents on absolute value of x
            sigma_noise = self.sigma_obs * np.exp(-1 * np.abs(self.x) / self.noise_const)
        else:
            sigma_noise = self.sigma_obs

        L = np.linalg.cholesky(K + np.multiply(np.square(sigma_noise), np.eye(K.shape[0])))
        alpha = np.linalg.lstsq(L.transpose(), np.linalg.lstsq(L, self.y - y_bias)[0])[0]
        f = np.dot(K_test, alpha)

        self.f = f + y_bias
        self.v = np.linalg.lstsq(L, K_test.transpose())[0]
        self.v_f = self.cov_kernel(self.x_test, self.x_test) - np.dot(self.v.transpose(), self.v)

        self.lp = -1 / 2 * (self.y - y_bias).transpose() * alpha - np.sum(np.log(K)) - self.x.shape[0] / 2 * np.log(2 * np.pi)

        return self.f, self.v_f, self.lp

    def sample(self, n_samples):
        """
        Sample at x_test
        :param n_samples: number of repeated samples to generate
        :return:
        y_test: generated samples (k, n_samples)
        """
        self.y_test = np.random.multivariate_normal(np.squeeze(self.f), self.v_f, n_samples).T

        return self.y_test


    def cov_kernel(self, x1, x2):

        K = np.zeros((x1.shape[0], x2.shape[0]))

        for i in range(x1.shape[0]):
            dist = x1[i, :] - x2

            if not self.if_stat_kernel:
                len_scale = np.abs(x1[i, :])
            else:
                len_scale = 1
            # Sum over the dimension d
            K[i, :] = self.scale * np.exp(-0.5 * np.square(dist / (self.sigma_kernel * np.sqrt(len_scale) + self.bias))).transpose()

        return K
