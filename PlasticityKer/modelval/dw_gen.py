"""
    Generate dw according to real data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns
import tensorflow as tf
from modelval import pairptl, network, trainer, dataset
from modelval.ArbDataGen import arb_w_gen

def STDP_dw_gen(n_neighbors=3, df=None):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
    data1 = data[data['ptl_idx'] == 1]
    x = np.array(data1['dt1']).reshape(-1, 1)
    y = np.array(data1['dw_mean']).reshape(-1, 1)

    # Generate dt1 if dt_list is None
    # Insert values for STDP
    if df is None:
        dt = np.arange(-100, 100, 2)
        data1_gen = pd.DataFrame(data=None, columns=list(data.columns))
        for i in range(len(dt)):
            new_try1 = data1.iloc[0]
            new_try1['dt1'] = dt[i]
            data1_gen = data1_gen.append(new_try1, ignore_index=True)

        df = data1_gen

    # Use K nearest neighbors to estimate the mean value of a given dt1
    weights = 'uniform'

    Kn_reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    Kn_reg.fit(x, y)
    y_pred = Kn_reg.predict(np.array(df['dt1']).reshape(-1, 1))

    return df, y_pred

def triplet_dw_gen(dt=None):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
    data2 = data[data['ptl_idx'] == 2]
    data4 = data[data['ptl_idx'] == 4]

    if dt is None:
        dt = np.array([-10, -5, 0, 5, 10]).reshape(-1, 1)

    data2_gen = pd.DataFrame(data=None, columns=list(data.columns))

    for i in range(len(dt)):
        new_try2 = data2.iloc[0]
        if dt[i] < 0:
            new_try2['dt1'] = dt[i]
            new_try2['dt2'] = dt[i]
            new_try2['dw_mean'] = data2[data2['dt1'] < 0]['dw_mean'].mean()
        if dt[i] == 0:
            new_try2['dt1'] = dt[i]
            new_try2['dt2'] = dt[i]
            new_try2['dw_mean'] = 60
        elif dt[i] > 0:
            new_try2['dt1'] = dt[i]
            new_try2['dt2'] = dt[i]
            new_try2['dw_mean'] = data2[data2['dt1'] > 0]['dw_mean'].mean()

        data2_gen = data2_gen.append(new_try2, ignore_index=True)

    targets2 = np.array(data2_gen['dw_mean']).reshape(-1, 1)

    data4_gen = data4
    targets4 = np.array(data3['dw_mean']).reshape(-1, 1)

    data_gen = pd.concat([data2_gen, data4_gen])
    targets = np.concatenate([targets2, targets4])

    return data_gen, targets
