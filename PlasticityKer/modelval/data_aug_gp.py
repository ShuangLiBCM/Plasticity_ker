"""
Generate STDP data, triplet and quadruplet data with Gaussian Process
"""
import numpy as np
import pandas as pd
from modelval import gp_regressor
import pdb


def stdp_gp(random_state=0, test_fold=0, **params):
    """
    Generate augmented STDP data through sample from the GP regressor with designated parameter
    :param random_state:
    :param params: Enter as keyword argument to the GP regressor
    :return:
    """

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data1 = data[data['ptl_idx'] == 1]

    train_index, test_index = train_test_split(data1['dt1'].shape[0], test_fold=test_fold, random_state=random_state)

    x_train = data1['dt1'].values[train_index]
    y_train = data1['dw_mean'].values[train_index]
    x_test = data1['dt1'].values[test_index]
    y_test = data1['dw_mean'].values[test_index]

    # Fit the training data with Gaussian process
    std_stdp = data1[np.abs(data1['dt1']) > 50]['dw_mean'].std()
    x = x_train.reshape(-1, 1)/100
    y = y_train.reshape(-1, 1)/std_stdp

    x_aug = np.linspace(np.min(x), np.max(x), 200).reshape(-1, 1)

    # Use Gaussian regressor that have been validated using hyper-parameter
    gp_rg = gp_regressor.GP_regressor(x, y, x_aug, **params)
    f_mean, v_f, lp = gp_rg.fit(y_bias=0)

    f_std = np.sqrt(v_f.transpose().diagonal()).reshape(-1, 1)
    # plt.plot(x, y, 'o', label='Raw_train_data')

    # Sample from the modified distribution

    # cov_gen = v_f + np.dot(noise_sigma.T, np.eye(noise_sigma.shape[0]))
    f_samp = np.diagonal(gp_rg.sample(x_aug.shape[0])) * std_stdp

    #f_samp = np.zeros(f_mean.shape)

    #for i in range(len(f_samp)):
    #    np.random.seed(i)
    #    scale = 5 * np.exp(-1 * np.abs(x_aug[i]) / 81.1)
    #    noise = np.random.normal(loc=0, scale=scale, size=1)
    #    f_samp[i] = f_mean[i] + noise

    # plt.plot(x_aug, f_samp, 'ro', label='Sampled data')
    # plt.plot(x_aug, f, label='Mean function')
    #
    # plt.legend(loc='upper left')
    return x_aug * 100, f_samp.reshape(-1, 1), x_test, y_test.reshape(-1, 1), f_mean * std_stdp


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


def quad_gp(random_state=100, test_fold=0, **params):
    """
    Generate augmented Quadruplet data through sample from the GP regressor with designated parameter
    :param random_state:
    :param params:
    :return:
    """

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data3 = data[data['ptl_idx'] == 3]

    # Split into training and testing set
    train_index, test_index = train_test_split(data3['dt2'].shape[0], n_folds=5,
                                               test_fold=test_fold, random_state=random_state)

    x_train = data3['dt2'].values[train_index]
    y_train = data3['dw_mean'].values[train_index]
    x_test = data3['dt2'].values[test_index]
    y_test = data3['dw_mean'].values[test_index]

    std_quad = data3[(data3['dt2'] < -50)]['dw_mean'].std()

    x = np.array(data3['dt2']).reshape(-1, 1) / 100
    y = np.array(data3['dw_mean']).reshape(-1, 1) / std_quad

    # Find the min and max of x for negative dt and postive dt
    x_neg_min, x_neg_max = np.min(data3[data3['dt2'] < 0]['dt2']) / 100, np.max(data3[data3['dt2'] < 0]['dt2']) / 100
    x_posi_min, x_posi_max = np.min(data3[data3['dt2'] > 0]['dt2']) / 100, np.max(data3[data3['dt2'] > 0]['dt2']) / 100

    x_aug = np.concatenate([np.linspace(x_neg_min, x_neg_max, 100),
                            np.linspace(x_posi_min, x_posi_max, 100)]).reshape(-1, 1)

    # Use Gaussian regressor that have been validated using hyper-parameter
    gp_rg = gp_regressor.GP_regressor(x, y, x_aug, **params)
    f_mean, v_f, lp = gp_rg.fit(y_bias=0)

    # Sample from the gp regression
    f_samp = np.diagonal(gp_rg.sample(x_aug.shape[0])) * std_quad

    """
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
    """
    return x_aug * 100, f_samp.reshape(-1, 1), x_test, y_test.reshape(-1, 1), f_mean * std_quad


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


def triplet_dw_gen(random_state=10, test_fold=0, n_samples=20):

    # Obtain data from triplet protocol

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
    data2 = data[data['ptl_idx'] == 2]

    # Split into training and testing set
    train_index, test_index = train_test_split(data2.shape[0], n_folds=5,
                                               test_fold=test_fold, random_state=random_state)

    data2_train = data2.iloc[train_index]
    data2_test = data2.iloc[test_index]

    dt_choices = np.array(data2_train['dt1'].value_counts().index)

    # Sample from the mean value of the training data to fill into the data frame
    data2_train_gen = pd.DataFrame(data=None, columns=list(data2.columns))

    for i in range(len(dt_choices)):
        new_entry = data2[data2['dt1'] == dt_choices[i]].iloc[0]
        data_mean = data2_train[data2_train['dt1'] == dt_choices[i]]['dw_mean'].mean()
        data_size = len(data2_train[data2_train['dt1'] == dt_choices[i]])
        data_scale = data2_train[data2_train['dt1'] == dt_choices[i]]['dw_mean'].std()/np.sqrt(data_size)
        if n_samples > 1:
            sample = np.random.normal(loc=data_mean, scale=data_scale, size=n_samples)

            for j in range(n_samples):
                new_entry['dw_mean'] = sample[j]
                data2_train_gen = data2_train_gen.append(new_entry, ignore_index=True)
        else:
            sample = data_mean
            new_entry['dw_mean'] = sample
            data2_train_gen = data2_train_gen.append(new_entry, ignore_index=True)

    y_train2 = np.array(data2_train_gen['dw_mean']).reshape(-1, 1)

    # Use the raw data as the testing data
    data2_test_gen = data2_test
    y_test2 = np.array(data2_test_gen['dw_mean']).reshape(-1, 1)

    # Sample from the raw ptl 4 data as the training data
    data4 = data[data['ptl_idx'] == 4]
    data4_train_gen = pd.DataFrame(data=None, columns=list(data2.columns))

    dt_choices1 = np.array(data4['dt1'].value_counts().index)
    dt_choices2 = np.array(data4['dt2'].value_counts().index)

    for i in range(len(dt_choices1)):
        new_entry = data4[(data4['dt1'] == dt_choices1[i]) & (data4['dt2'] == dt_choices2[i])]
        data_mean = new_entry['dw_mean']
        data_scale = new_entry['dw_ste']

        if n_samples > 1:
            sample = np.random.normal(loc=data_mean, scale=data_scale, size=n_samples)
            for j in range(n_samples):
                new_entry['dw_mean'] = sample[j]
                data4_train_gen = data4_train_gen.append(new_entry, ignore_index=True)
        else:
            sample = data_mean
            new_entry['dw_mean'] = sample
            data4_train_gen = data4_train_gen.append(new_entry, ignore_index=True)

    y_train4 = np.array(data4_train_gen['dw_mean']).reshape(-1, 1)

    data_train_gen = pd.concat([data2_train_gen, data4_train_gen])
    y_train = np.concatenate([y_train2, y_train4])

    data_test_gen = pd.concat([data2_test_gen, data4])
    y_test = np.concatenate([y_test2, data4['dw_mean'].reshape(-1, 1)])

    return data_train_gen, y_train.reshape(-1, 1), data_test_gen, y_test.reshape(-1, 1)


def smooth(x, width=10, width_list=None):
    y = np.zeros(x.shape)
    if width_list is None:
        for i in range(x.shape[0]):
            y[i] = np.mean(x[np.max([0, i-int(width/2)]):np.min([x.shape[0], i+int(width/2)])])
    else:
        for i in range(x.shape[0]):
            y[i] = np.mean(x[np.max([0, i-int(width_list[i])]):np.min([x.shape[0], i+int(width_list[i])])])
    return y


def train_test_split(total_len, n_folds=5, test_fold=0, random_state=0):

    np.random.seed(random_state)
    index = np.random.permutation(range(total_len))
    test_size = int(total_len/n_folds)

    test_index = index[test_fold * test_size: (test_fold+1)*test_size]
    train_index = np.setdiff1d(index, test_index)


    return train_index, test_index