"""
Generate STDP data with Gaussian Process
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelval import gp_regressor


def stdp_gp():

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data1 = data[data['ptl_idx'] == 1]

    x_train, x_test, y_train, y_test = train_test_split(data1['dt1'], data1['dw_mean'], test_size=0.2, random_state=0)

    # Fit the training data with Gaussian process
    x = x_train.reshape(-1, 1)
    y = y_train.reshape(-1, 1)
    x_aug = np.linspace(-100, 100, 200).reshape(-1, 1)

    # Use Gaussin regressor that have been validated using hyperparameter
    gp_rg = gp_regressor.GP_regressor(x, y, x_aug, sigma_kernel=1.68, scale=20.0, bias=4.5, sigma_obs=5.0,
                                      noise_const=81.1, if_stat_kernel=False, if_stat_noise=False)
    f_mean, v_f, lp = gp_rg.fit()

    f_std = np.sqrt(v_f.transpose().diagonal()).reshape(-1, 1)
    # plt.plot(x, y, 'o', label='Raw_train_data')

    # Sample from the modified distribution

    # cov_gen = v_f + np.dot(noise_sigma.T, np.eye(noise_sigma.shape[0]))
    # f_samp = gp_rg.sample(10, cov=cov_gen)

    scale = np.zeros(f_mean.shape)
    noise = np.zeros(f_mean.shape)
    f_samp = np.zeros(f_mean.shape)

    for i in range(len(f_samp)):
        np.random.seed(i)
        scale[i] = 5 * np.exp(-1 * np.abs(x_aug[i]) / 81.1)
        noise[i] = np.random.normal(loc=0, scale=scale[i], size=1)
        f_samp[i] = f_mean[i] + noise[i]

    # plt.plot(x_aug, f_samp, 'ro', label='Sampled data')
    # plt.plot(x_aug, f, label='Mean function')
    #
    # plt.legend(loc='upper left')
    return x_aug, f_samp, f_mean, f_std, x_test, y_test


def STDP_dw_gen(dt, df_ori=None):
    """
    put dt into data frame
    """
    if df_ori is None:
        data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
        data1 = data[data['ptl_idx'] == 1]
        x = np.array(data1['dt1']).reshape(-1, 1)
        y = np.array(data1['dw_mean']).reshape(-1, 1)
    else:
        data = df_ori
        data1 = data[data['ptl_idx'] == 1]
        x = np.array(data1['dt1']).reshape(-1, 1)
        y = np.array(data1['dw_mean']).reshape(-1, 1)

    # Insert values for STDP
    data1_gen = pd.DataFrame(data=None, columns=list(data.columns))
    for i in range(len(dt)):
        new_try1 = data1.iloc[0]
        new_try1['dt1'] = dt[i]
        data1_gen = data1_gen.append(new_try1, ignore_index=True)

    df = data1_gen

    return df