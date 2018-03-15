"""
Implement the Gaussian process with non-stationary kernel and noise
"""
import numpy as np
from scipy.stats import multivariate_normal
import pdb

class GP_regressor(object):

    def __init__(self, x=None, y=None, x_test=None, length_scale=1, amp_kernel=1, power_sc=0, scale_sc=0,
                 sigma_noise=0, noise_sc=0):
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
        self.sigma_obs = sigma_noise
        self.sigma_kernel = length_scale
        self.noise_const = noise_sc
        self.scale = amp_kernel
        self.power_sc = power_sc
        self.scale_sc = scale_sc
        self.bias = np.min(np.abs(np.diff(np.hstack(x_test))))

    def fit(self, y_bias=None):
        """
        Perform Gaussian Regression on training data and generate prediction for testing data
        y_bias: Enter zero if assume zeros mean, else None
        :return:
        f: mean of the posterior distribution over function
        v_f: covariance of the posterior distribution over the function
        lp: log likelihood of posterior distribution of function over training data


        Refer to book "Gaussian Processes for Machine Learning" algorithm 2.3

        """

        if (self.x is not None) & (self.y is not None):

            if y_bias is None:
                y_bias = np.squeeze(np.mean(self.y, axis=0))  # Assume zeros mean, remove bias first, add it back later

            K = self.cov_kernel(self.x, self.x)
            K_test = self.cov_kernel(self.x_test, self.x)

            if self.noise_const > 0:
                # Implement non-stationary noise that dependents on absolute value of x
                sigma_noise = self.sigma_obs / np.exp(-1 * np.abs(1-np.abs(self.x)) / self.noise_const)
            else:
                sigma_noise = self.sigma_obs

            L = np.linalg.cholesky(K + np.multiply(np.square(sigma_noise), np.eye(K.shape[0])))
            alpha = np.linalg.lstsq(L.transpose(), np.linalg.lstsq(L, self.y - y_bias)[0])[0]
            f = np.dot(K_test, alpha)

            self.f = f + y_bias
            self.v = np.linalg.lstsq(L, K_test.transpose())[0]
            self.v_f = self.cov_kernel(self.x_test, self.x_test) - np.dot(self.v.transpose(), self.v)

            self.lp = -1 / 2 * (self.y - y_bias).transpose() * alpha - np.sum(np.log(K)) - self.x.shape[0] / 2 * np.log(2 * np.pi)

        else:  # Return prior if no data is entered
            if y_bias is None:
                self.f = np.zeros(self.x_test.shape)
            else:
                self.f = y_bias

            K = self.cov_kernel()
            self.v_f = K

            self.lp = multivariate_normal.pdf(x=self.x_test, mean=np.squeeze(self.f), cov=self.v_f, allow_singular=True)


        return self.f, self.v_f, self.lp

    def cov_kernel(self, x1=None, x2=None):

        if (x1 is None) | (x2 is None):
            x1 = self.x_test
            x2 = self.x_test

        K = np.zeros((x1.shape[0], x2.shape[0]))

        for i in range(x1.shape[0]):
            dist = x1[i, :] - x2

            # Implement non-stationary through length_scale
            if self.power_sc > 0:
                len_scale_sc = np.power(np.abs(x1[i, :]) + np.abs(x2), self.power_sc)
            else:
                len_scale_sc = 1
                self.bias = 0

            # Sum over the dimension d
            K[i, :] = np.multiply(self.scale, np.exp(-0.5 * np.square(dist / (self.sigma_kernel * len_scale_sc + self.bias)))).transpose()

        return K

    def sample(self, n_samples, cov=None, sample_reso=1, seed=0):
        """
        Sample at x_test
        :param n_samples: number of repeated samples to generate
        :param cov: specify the covariance matrix to generate data from
        :param sample_reso: number of points to drawn from each function
        :return:
        y_test: generated samples (k, n_samples)
        """
        y_test = []
        # if cov is None:
        #     for j in range(n_samples):
        #         sample = np.zeros(self.f.shape[0])
        #         for i in range(self.f.shape[0]):
        #             np.random.seed(j*i+i)
        #             sample_tmp = np.random.multivariate_normal(np.squeeze(self.f), self.v_f, 1).T
        #             sample[i] = sample_tmp[i]
        #         y_test.append(sample)
        if cov is None:
            for i in range(n_samples):
                np.random.seed(seed+i)
                y_test.append(np.random.multivariate_normal(np.squeeze(self.f), self.v_f, 1))
        else:
            for j in range(n_samples):
                sample = np.zeros(self.f.shape[0])
                for i in range(self.f.shape[0]):
                    np.random.seed(j*i+i)
                    sample_tmp = np.random.multivariate_normal(np.squeeze(self.f), cov, sample_reso).T
                    sample[i] = sample_tmp[i]
                y_test.append(sample)

        return np.vstack(y_test).transpose()