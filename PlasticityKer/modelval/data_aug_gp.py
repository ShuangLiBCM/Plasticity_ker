"""
Generate STDP data with Gaussian Process
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelval import gp_regressor


def stdp_gp(random_state=0):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data1 = data[data['ptl_idx'] == 1]

    x_train, x_test, y_train, y_test = train_test_split(data1['dt1'], data1['dw_mean'], test_size=0.2, random_state=random_state)

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

    f_samp = np.zeros(f_mean.shape)

    for i in range(len(f_samp)):
        np.random.seed(i)
        scale = 5 * np.exp(-1 * np.abs(x_aug[i]) / 81.1)
        noise = np.random.normal(loc=0, scale=scale, size=1)
        f_samp[i] = f_mean[i] + noise

    # plt.plot(x_aug, f_samp, 'ro', label='Sampled data')
    # plt.plot(x_aug, f, label='Mean function')
    #
    # plt.legend(loc='upper left')
    return x_aug, f_samp, x_test, y_test


def STDP_dw_gen(dt, df_ori=None):
    """
    put dt into data frame
    """
    if df_ori is None:
        data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
        data1 = data[data['ptl_idx'] == 1]
    else:
        data = df_ori
        data1 = data[data['ptl_idx'] == 1]

    # Insert values for STDP
    data1_gen = pd.DataFrame(data=None, columns=list(data.columns))
    for i in range(len(dt)):
        new_try1 = data1.iloc[0]
        new_try1['dt1'] = dt[i]
        data1_gen = data1_gen.append(new_try1, ignore_index=True)

    df = data1_gen

    return df

def quad_gp(random_state=0):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data3 = data[data['ptl_idx'] == 3]

    x_train, x_test, y_train, y_test = train_test_split(data3['dt2'].values, data3['dw_mean'].values, test_size=0.2,
                                                        random_state=random_state)

    x_r = x_train[np.where(x_train > 0)[0]].reshape(-1, 1)
    y_r = y_train[np.where(x_train > 0)[0]].reshape(-1, 1)
    x_test_r = np.linspace(np.min(x_r), 120, 120).reshape(-1, 1)

    x_l = x_train[np.where(x_train < 0)[0]].reshape(-1, 1)
    y_l = y_train[np.where(x_train < 0)[0]].reshape(-1, 1)
    x_test_l = np.linspace(-120, np.max(x_l), 120).reshape(-1, 1)

    gp_rg = gp_regressor.GP_regressor(x_r, y_r, x_test_r, sigma_kernel=1.9, scale=5, bias=2.78, sigma_obs=3.0,
                                      noise_const=96.7, if_stat_kernel=False, if_stat_noise=False)
    f_r_mean, v_f_r, lp = gp_rg.fit()
    std_r = np.sqrt(v_f_r.transpose().diagonal()).reshape(-1, 1)

    gp_rg = gp_regressor.GP_regressor(x_l, y_l, x_test_l, sigma_kernel=1.9, scale=5, bias=2.78, sigma_obs=3.0,
                                      noise_const=96.7, if_stat_kernel=False, if_stat_noise=False)
    f_l_mean, v_f_l, lp = gp_rg.fit()
    std_l = np.sqrt(v_f_l.transpose().diagonal()).reshape(-1, 1)

    # Sample from the gp regression
    f_l_samp = np.zeros(f_l_mean.shape)
    f_r_samp = np.zeros(f_r_mean.shape)

    for i in range(len(f_l_mean)):
        np.random.seed(i)
        scale = 5 * np.exp(-1 * np.abs(x_test_l[i]) / 96.7)
        noise = np.random.normal(loc=0, scale=scale, size=1)
        f_l_samp[i] = f_l_mean[i] + noise

    for i in range(len(f_r_mean)):
        np.random.seed(i)
        scale = 5 * np.exp(-1 * np.abs(x_test_r[i]) / 96.7)
        noise = np.random.normal(loc=0, scale=scale, size=1)
        f_r_samp[i] = f_r_mean[i] + noise

    return x_test_r, f_r_samp, x_test_l, f_l_samp, x_test, y_test

def quad_dw_gen(dt, df_ori=None):
    """
    put dt into data frame
    """
    if df_ori is None:
        data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
        data3 = data[data['ptl_idx'] == 3]
    else:
        data = df_ori
        data3 = data[data['ptl_idx'] == 3]

    # Generate dt1 if dt_list is None
    # Insert values for STDP
    data3_gen = pd.DataFrame(data=None, columns=list(data.columns))
    for i in range(len(dt)):
        new_try3 = data3.iloc[0]
        new_try3['dt2'] = dt[i]
        data3_gen = data3_gen.append(new_try3, ignore_index=True)

        df = data3_gen

    return df