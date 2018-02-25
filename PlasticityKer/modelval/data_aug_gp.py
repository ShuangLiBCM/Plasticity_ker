"""
Generate STDP data, triplet and quadruplet data with Gaussian Process
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelval import gp_regressor
import pdb

def stdp_gp(random_state=0, **params):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data1 = data[data['ptl_idx'] == 1]

    x_train, x_test, y_train, y_test = train_test_split(data1['dt1'], data1['dw_mean'], test_size=0.2, random_state=random_state)

    # Fit the training data with Gaussian process
    x = x_train.reshape(-1, 1)
    y = y_train.reshape(-1, 1)
    scaler = data1[np.abs(data1['dt1'])>50]['dw_mean'].std()
    y = y/scaler
    x_aug = np.linspace(np.min(x),np.max(x),200).reshape(-1,1)

    # Use Gaussin regressor that have been validated using hyperparameter
    gp_rg = gp_regressor.GP_regressor(x, y, x_aug, **params)
    f_mean, v_f, lp = gp_rg.fit()
    
    f_std = np.sqrt(v_f.transpose().diagonal()).reshape(-1, 1)
    # plt.plot(x, y, 'o', label='Raw_train_data')

    # Sample from the modified distribution

    # cov_gen = v_f + np.dot(noise_sigma.T, np.eye(noise_sigma.shape[0]))
    f_samp = gp_rg.sample(1)*scaler

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
    return x_aug, f_samp.reshape(-1,1), x_test, y_test.reshape(-1,1)

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

def quad_gp(random_state=0, **params):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data3 = data[data['ptl_idx'] == 3]

    x_train, x_test, y_train, y_test = train_test_split(data3['dt2'].values, data3['dw_mean'].values, test_size=0.2,
                                                        random_state=random_state)
    
    scaler = data3[np.abs(data3['dt2'])>100]['dw_mean'].std()
    y_train = y_train/scaler
    
    x_r = x_train[np.where(x_train > 0)[0]].reshape(-1, 1)
    y_r = y_train[np.where(x_train > 0)[0]].reshape(-1, 1)
    x_test_r = np.linspace(np.min(x_r), 120, 120).reshape(-1, 1)

    x_l = x_train[np.where(x_train < 0)[0]].reshape(-1, 1)
    y_l = y_train[np.where(x_train < 0)[0]].reshape(-1, 1)
    x_test_l = np.linspace(-120, np.max(x_l), 120).reshape(-1, 1)

    gp_rg_r = gp_regressor.GP_regressor(x_r, y_r, x_test_r, **params)
    f_r_mean, v_f_r, lp = gp_rg_r.fit()
    std_r = np.sqrt(v_f_r.transpose().diagonal()).reshape(-1, 1)

    gp_rg_l = gp_regressor.GP_regressor(x_l, y_l, x_test_l, **params)
    f_l_mean, v_f_l, lp = gp_rg_l.fit()
    std_l = np.sqrt(v_f_l.transpose().diagonal()).reshape(-1, 1)

    # Sample from the gp regression
    f_l_samp = gp_rg_l.sample(1)*scaler
    f_r_samp = gp_rg_r.sample(1)*scaler
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
    return x_test_r, f_r_samp.reshape(-1,1), x_test_l, f_l_samp.reshape(-1,1), x_test, y_test.reshape(-1,1)

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

def triplet_dw_gen(dt=None, n_samples=20):

    # Obtain data from triplet protocol

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
    data2 = data[data['ptl_idx'] == 2]

    # Split the raw ptl2 data into 80%, 20% as training and testing data
    np.random.seed(0)
    data2_idx = np.random.permutation(len(data2))
    train_len = int(len(data2)*0.8)
    data2_train = data2.iloc[data2_idx[:train_len]]
    data2_test = data2.iloc[data2_idx[train_len:]]
    
    dt_choices = np.array(data2_train['dt1'].value_counts().index)
    
    # Sample from the mean value of the training data to fill into the data frame
    data2_train_gen = pd.DataFrame(data=None, columns=list(data2.columns))
    for i in range(len(dt_choices)):
        new_entry = data2[data2['dt1'] == dt_choices[i]].iloc[0]
        sample = np.random.normal(loc=data2_train[data2_train['dt1']==dt_choices[i]]['dw_mean'].mean(), scale=data2_train[data2_train['dt1']==dt_choices[i]]['dw_mean'].std()/np.sqrt(len(data2_train[data2_train['dt1']==dt_choices[i]])), size=n_samples) 
        for j in range(n_samples):
            new_entry['dt1']
            new_entry['dw_mean'] = sample[j]
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
    
    # pdb.set_trace()
    for i in range(len(dt_choices1)):
        new_entry = data4[(data4['dt1'] == dt_choices1[i])&(data4['dt2'] == dt_choices2[i])]
        sample = np.random.normal(loc=new_entry['dw_mean'], scale=new_entry['dw_ste'], size=n_samples)
        for j in range(n_samples):
            new_entry['dw_mean'] = sample[j]
            data4_train_gen = data4_train_gen.append(new_entry, ignore_index=True)

    y_train4 = np.array(data4_train_gen['dw_mean']).reshape(-1, 1)
                                  
    data_train_gen = pd.concat([data2_train_gen, data4_train_gen])
    y_train = np.concatenate([y_train2, y_train4])

    data_test_gen = pd.concat([data2_test_gen, data4])
    y_test = np.concatenate([y_test2, data4['dw_mean'].reshape(-1,1)])
    
    return data_train_gen, y_train.reshape(-1,1), data_test_gen, y_test.reshape(-1,1)