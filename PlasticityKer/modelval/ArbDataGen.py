"""
    Functions for generating spike train from pairing ptl
"""
import numpy as np
from modelval import pairptl, network, trainer, data_aug_gp
from sklearn.model_selection import KFold
import pandas as pd

import pdb

def arb_spk_gen(ptl, spk_reso, spk_len=None, if_noise=1, seed=None):
    """
    Generate a single set of pre-spike, post-spike trains and target given the ptl
    -------------------------------
    :param ptl: PairPtl object
    :param spk_reso: Sampling resolution in ms, should be consistent with kernel sampling resolution
    :param spk_len: length of the longest spike train
    :param if_noise: Whether to introduce noise during the spike generation
    :param seed: seed for the random generator
    :return:
    """

    # Define resolution of spike train
    train_len = int(ptl.ptl_occ / ptl.ptl_freq)

    # Define length of the spike train
    if spk_len is None:
        spk_len = train_len//spk_reso * 1000     # spk_reso is in msec

    spk_pair = np.zeros((spk_len+2, 2))

    # Convert protocl number into dummy variable
    ptl_indicator = np.zeros((4))
    ptl_indicator[int(ptl.ptl_idx-1)] = 1

    spk_pair[-2:,0] = ptl_indicator[:2]
    spk_pair[-2:,1] = ptl_indicator[2:]

    rep_interval = int(np.floor(1000 / ptl.ptl_freq / spk_reso))
    rep_num = int(np.floor(train_len * ptl.ptl_freq))

    # Consider the max dt = 120 ms
    min_bef = int(120//spk_reso + 1)

    if seed is not None:
        np.random.seed(seed)

    # pre-post: 1-1
    if (int(ptl.pre_spk_num) == 1) & (int(ptl.post_spk_num) == 1):

        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        # Obtain time index of spike
        spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
        mean_dt = int(ptl.dt1 / spk_reso)
        # Generate noise
        if if_noise:
            # Choose mean_dt as 2 * std
            between_noise = np.random.normal(loc=0.0, scale=np.abs(mean_dt/2), size=spk_time_pre.shape).astype(int)

            for i in range(len(between_noise)):
                if np.abs(between_noise[i]) >= np.abs(mean_dt):
                    between_noise[i] = 0

            spk_time_post = spk_time_pre + mean_dt + between_noise   # dt1, dt2, dt3 in ms
        else:
            spk_time_post = spk_time_pre + mean_dt  # dt1, dt2, dt3 in ms

    # pre-post: 2-1
    elif (int(ptl.pre_spk_num) == 2) & (int(ptl.post_spk_num) == 1):

        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        mean_dt1 = int(ptl.dt1 / spk_reso)
        mean_dt2 = int(ptl.dt2 / spk_reso)
        # Obtain time index of spike
        if if_noise:
            spk_time_post = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            between_noise1 = np.random.normal(loc=0.0, scale=np.abs(mean_dt1/2), size=spk_time_post.shape).astype(int)

            for i in range(len(between_noise1)):
                # triplet data is limited, prevent the possibility for over-interpolation, especially for small dt.
                if (np.abs(between_noise1[i]) >= np.min([np.abs(mean_dt1), np.abs(mean_dt2)]))\
                        | (np.abs(mean_dt1 + between_noise1[i]) <= 1) | (np.abs(mean_dt2 + between_noise1[i]) <= 1):
                    between_noise1[i] = 0

            # For triplet model, create symmetric noise
            spk_time_pre1 = spk_time_post + mean_dt1 + between_noise1
            spk_time_pre2 = spk_time_post + mean_dt2 - between_noise1
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
        else:
            spk_time_post = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            spk_time_pre1 = spk_time_post + mean_dt1
            spk_time_pre2 = spk_time_post + mean_dt2
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
    # pre-post: 1-2
    elif (int(ptl.pre_spk_num) == 1) & (int(ptl.post_spk_num) == 2):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        mean_dt1 = int(ptl.dt1 / spk_reso)
        mean_dt2 = int(ptl.dt2 / spk_reso)
        # Obtain time index of spike
        if if_noise:
            spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            between_noise1 = np.random.normal(loc=0.0, scale=np.abs(mean_dt1/2), size=spk_time_pre.shape).astype(int)
            for i in range(len(between_noise1)):
                if (np.abs(between_noise1[i]) >= np.min([np.abs(mean_dt1), np.abs(mean_dt2)]))|(np.abs(mean_dt1 + between_noise1[i])<=1)|(np.abs(mean_dt2 + between_noise1[i])<=1):
                    between_noise1[i] = 0

            spk_time_post1 = spk_time_pre - mean_dt1 + between_noise1
            spk_time_post2 = spk_time_pre - mean_dt2 - between_noise1
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

        else:
            spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            spk_time_post1 = spk_time_pre - mean_dt1
            spk_time_post2 = spk_time_pre - mean_dt2
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

    # pre-post: 2-2
    elif (int(ptl.pre_spk_num) == 2) & (int(ptl.post_spk_num) == 2):
        spk_time_base1 = np.random.randint(min_bef, rep_interval - min_bef, 1)
        mean_dt = int(ptl.dt2/spk_reso)

        time_max = 90//spk_reso
        time_min = 5//spk_reso

        if if_noise:
            # mean_dt is large, choose a small value in case the variability is too large

            within_noise = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt/2), 2]), size=spk_time_base1.shape).astype(int)
            # Eliminate the possibilities for over-interpolation < 5 ms or > 50 ms, where data is limited.
            if (np.abs(within_noise) < np.abs(mean_dt)) & (np.abs(mean_dt) + within_noise > time_min) & \
                    (np.abs(mean_dt) + within_noise <= time_max):
                spk_time_base2 = spk_time_base1 + np.abs(mean_dt) + within_noise
            else:
                spk_time_base2 = spk_time_base1 + np.abs(mean_dt)
            #     if np.abs(mean_dt) < time_max:
            #         spk_time_base2 = spk_time_base1 + np.abs(mean_dt)
            #     else:
            #         spk_time_base2 = spk_time_base1 + time_max
            # print(spk_time_base2 - spk_time_base1)
        else:
            spk_time_base2 = spk_time_base1 + np.abs(mean_dt)
            #if np.abs(mean_dt) < time_max:
            #    spk_time_base2 = spk_time_base1 + np.abs(mean_dt)
            #else:
            #    spk_time_base2 = spk_time_base1 + time_max

        if ptl.dt2 == 0:
            ptl.dt2 = -1

        if ptl.dt2 < 0:  # Pre-post-post-pre
            spk_time_post1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_post2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])

            spk_time_pre1 = spk_time_post1 - int(ptl.dt1 / spk_reso)
            spk_time_pre2 = spk_time_post2 - int(ptl.dt3 / spk_reso)
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

        elif ptl.dt2 > 0:   # Post-pre-pre-post
            spk_time_pre1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_pre2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])

            spk_time_post1 = spk_time_pre1 - int(ptl.dt1 / spk_reso)
            spk_time_post2 = spk_time_pre2 - int(ptl.dt3 / spk_reso)
            # Obtain spike train
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

    # pre-post: 5-5
    elif (int(ptl.pre_spk_num) == 5) & (int(ptl.post_spk_num) == 5):
        isi = int(1000 / ptl.pre_spk_freq / spk_reso)
        spk_time_base = np.random.randint(min_bef, rep_interval - isi - min_bef, 1)
        spk_time_base5 = np.hstack([spk_time_base + isi * i for i in range(5)])

        spk_time_pre = np.sort(np.concatenate([spk_time_base5 + rep_interval * i for i in range(rep_num)]))
        mean_dt = int(ptl.dt1/spk_reso)
        spk_time_post = np.zeros(spk_time_pre.shape).astype(int)
        if if_noise:
            between_noise = np.random.normal(loc=0.0, scale=np.max([np.abs(mean_dt)/2, (isi-np.abs(mean_dt))/2]), size=spk_time_pre.shape).astype(int)
            for i in range(between_noise.shape[0]):
                if mean_dt < 0:
                    if mean_dt + between_noise[i] >= 0:
                        spk_time_post[i] = int(spk_time_pre[i])
                    elif -mean_dt - between_noise[i] >= isi:
                        spk_time_post[i] = int(spk_time_pre[i] - isi + 1)
                    else:
                        spk_time_post[i] = int(spk_time_pre[i] + mean_dt + between_noise[i])
                else:
                    if mean_dt + between_noise[i] < 0:
                        spk_time_post[i] = int(spk_time_pre[i] + 1)
                    elif mean_dt + between_noise[i] >= isi:
                        spk_time_post[i] = int(spk_time_pre[i] + isi)
                    else:
                        spk_time_post[i] = int(spk_time_pre[i] + mean_dt + between_noise[i])
        else:
            spk_time_post = spk_time_pre + mean_dt

        # Find the same spk_time_post and seperate them
        if (np.diff(spk_time_post) == 0).any():
            spk_time_post[np.where(np.diff(spk_time_post) == 0)[0]+1] = spk_time_post[np.where(np.diff(spk_time_post) == 0)[0]+1]+1
    else:
        print('Wrong ptl!')

    spk_time_pre[np.where(spk_time_pre > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    spk_time_post[np.where(spk_time_post > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    spk_pair[spk_time_pre.astype(int), 0] = 1
    spk_pair[spk_time_post.astype(int), 1] = 1

    return spk_time_pre, spk_time_post, spk_pair


def arb_w_gen(spk_pairs=None, df=None, ptl_list=None, kernel=None, spk_len=None, targets=None, aug_times=None,
              seed=None, net_type='triplet', if_noise=1, batch_size=1000):
    """
    Generate arbitrary target w with given spike trains, kernel and network
    three situations in total:
    1. generate spike pairs and use targets (spk_pairs is None, targets is not None)
    2. generate spike pairs and targets (spk_pairs is None, targets is None)
    3. use spike pairs and generate targets (spk_pairs is not None, targets is None)
    ------------------------------
    :param df: data frame with protocol information
    :param ptl_list: list of protocol indices
    :param kernel: kernel object used to generate the toy data
    :param spk_len: total length of spike train
    :param aug_times: list number of times to augment for each member of ptl_list
    :return:
    """

    if spk_pairs is None:
        spk_pairs = []
        target_gen = []

        if spk_len is None:
            spk_len = int(15 / 0.1 * 1000 / kernel.reso_kernel)   # The longest protocol

        for i in range(len(ptl_list)):
            data_ptl = df[df['ptl_idx'] == ptl_list[i]]

            if targets is not None:
                target_tmp = targets[(df['ptl_idx'] == ptl_list[i]).reshape(-1,1)]

            k = 0
            for j in range(len(data_ptl)):
                ptl_info = pairptl.PairPtl(*data_ptl.iloc[j])
                for s in range(aug_times[i]):
                    if seed:
                        seed_set = i * len(ptl_list) + j * len(data_ptl) + s
                    else:
                        seed_set = None
                    _, _, spk_pair = arb_spk_gen(ptl_info, kernel.reso_kernel, spk_len=spk_len, if_noise=if_noise, seed=seed_set)
                    spk_pairs.append(spk_pair)
                    if targets is not None:
                        target_gen.append(target_tmp[k])
                k += 1
        # Generate the spike data
        spk_pairs = np.array(spk_pairs)   # Check the dimension into  (m * n * 2)

    # Get the network used to generate prediction
    if targets is None:
        if net_type == 'pair':
            gen_pairnet = network.PairNet(kernel=kernel, ground_truth_init=1, n_input=spk_pairs.shape[1])
            # Send the network graph into trainer, and name of placeholder
            gen_pairnet_train = trainer.Trainer(gen_pairnet.prediction, gen_pairnet.prediction, input_name=gen_pairnet.inputs)

            # generate targets through evaluating the prediction of the network
            targets = gen_pairnet_train.evaluate(ops=gen_pairnet.prediction, inputs=spk_pairs)

        elif net_type == 'triplet':
            gen_tripnet = network.TripNet(kernel=kernel, ground_truth_init=1, n_input=spk_pairs.shape[1])

            # Send the network graph into trainer, and name of placeholder
            gen_tripnet_train = trainer.Trainer(gen_tripnet.prediction, gen_tripnet.prediction,
                                                input_name=gen_tripnet.inputs)

            # generate targets through evaluating the prediction of the network
            targets = np.zeros(spk_pairs.shape[0]).reshape(-1, 1)
            k = 0
            while k * batch_size < spk_pairs.shape[0]:
                targets[k*batch_size:(k+1)*batch_size] = gen_tripnet_train.evaluate(ops=gen_tripnet.prediction, inputs=spk_pairs[k*batch_size:(k+1)*batch_size,:,:])
                k += 1
            targets[k*batch_size:] = gen_tripnet_train.evaluate(ops=gen_tripnet.prediction, inputs=spk_pairs[k*batch_size:,:,:])

        else:
            print('Wrong network!!')
    else:
        targets = np.vstack(target_gen)

    return spk_pairs, targets


def data_Gen(data_type='hippocampus', data_aug='gp_mean', test_fold_num=0, vali_split=5):

    if data_type == 'hippocampus':

        if data_aug == 'raw_data':
            data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')
            data1 = data[data['ptl_idx'] == 1]

            train_index1, test_index = data_aug_gp.train_test_split(len(data1), test_fold=test_fold_num, random_state=2)
            data_stdp_test = data1.iloc[test_index]
            y_stdp_test = np.array(data_stdp_test['dw_mean']).reshape(-1, 1)
            data_stdp_train1 = data1.iloc[train_index1]

            kf_stdp = KFold(n_splits=5)

            data_stdp_train, data_stdp_vali, y_stdp_train, y_stdp_vali = [], [], [], []

            for train_index, test_index in kf_stdp.split(data_stdp_train1):
                data_stdp_train.append(data_stdp_train1.iloc[train_index])
                data_stdp_vali.append(data_stdp_train1.iloc[test_index])
                y_stdp_train.append(np.array(data_stdp_train1.iloc[train_index]['dw_mean']).reshape(-1, 1))
                y_stdp_vali.append(np.array(data_stdp_train1.iloc[test_index]['dw_mean']).reshape(-1, 1))

            # Generate data for STDP
            data3 = data[data['ptl_idx'] == 3]

            # Split into training, testing and validation set
            train_index1, test_index = data_aug_gp.train_test_split(len(data3), test_fold=test_fold_num, random_state=200)
            data_quad_test = data3.iloc[test_index]
            y_quad_test = np.array(data_quad_test['dw_mean']).reshape(-1, 1)
            data_quad_train1 = data3.iloc[train_index1]

            kf_quad = KFold(n_splits=5)

            data_quad_train, data_quad_vali, y_quad_train, y_quad_vali = [], [], [], []

            for train_index, test_index in kf_quad.split(data_quad_train1):
                data_quad_train.append(data_quad_train1.iloc[train_index])
                data_quad_vali.append(data_quad_train1.iloc[test_index])
                y_quad_train.append(np.array(data_quad_train1.iloc[train_index]['dw_mean']).reshape(-1, 1))
                y_quad_vali.append(np.array(data_quad_train1.iloc[test_index]['dw_mean']).reshape(-1, 1))

            data_trip, y_trip, data_trip_test, y_trip_test = data_aug_gp.triplet_dw_gen(test_fold=test_fold_num,
                                                                                        n_samples=1)
            # Split the triplet data into training and validation
            kf_triplet = KFold(n_splits=vali_split, shuffle=True, random_state=456)
            data_trip_train, data_trip_vali, y_trip_train, y_trip_vali = [], [], [], []

            for train_index, vali_index in kf_triplet.split(data_trip):
                data_trip_train.append(data_trip.iloc[train_index])
                data_trip_vali.append(data_trip.iloc[vali_index])
                y_trip_train.append(y_trip[train_index])
                y_trip_vali.append(y_trip[vali_index])

            # Combine the shape

            data_gen_train, data_gen_vali, data_gen_test, y_train, y_vali, y_test = [], [], [], [], [], []

            for i in range(vali_split):

                data_gen_train.append(pd.concat([data_stdp_train[i], data_trip_train[i], data_quad_train[i]], axis=0))
                data_gen_vali.append(pd.concat([data_stdp_vali[i], data_trip_vali[i], data_quad_vali[i]], axis=0))

                y_train.append(np.concatenate([y_stdp_train[i], y_trip_train[i], y_quad_train[i]]).reshape(-1, 1))
                y_vali.append(np.concatenate([y_stdp_vali[i], y_trip_vali[i], y_quad_vali[i]]).reshape(-1, 1))

            data_gen_test.append(pd.concat([data_stdp_test, data_trip_test, data_quad_test], axis=0))
            y_test = np.concatenate([y_stdp_test, y_trip_test, y_quad_test]).reshape(-1, 1)

        else:
            # Generate data for STDP protocol
            params = {'amp_kernel': 1,
                      'length_scale': 0.1,
                      'power_sc': 0.2,
                      'sigma_noise': 1.0}

            if data_aug == 'gp_mean':
                x_stdp, _, x_stdp_test, y_stdp_test, y_stdp = data_aug_gp.stdp_gp(test_fold=test_fold_num, **params)
            elif data_aug == 'gp_samp':
                x_stdp, y_stdp, x_stdp_test, y_stdp_test, _ = data_aug_gp.stdp_gp(test_fold=test_fold_num, **params)

            kf_stdp = KFold(n_splits=vali_split, shuffle=True, random_state=12)

            x_stdp_train, x_stdp_vali, y_stdp_train, y_stdp_vali = [], [], [], []

            for train_index, vali_index in kf_stdp.split(x_stdp):
                x_stdp_train.append(x_stdp[train_index])
                y_stdp_train.append(y_stdp[train_index])
                x_stdp_vali.append(x_stdp[vali_index])
                y_stdp_vali.append(y_stdp[vali_index])

            # Generate data for quadruplet protocol
            params = {'amp_kernel': 1,
                      'length_scale': 0.45,
                      'power_sc': 0.4,
                      'sigma_noise': 1.2}

            if data_aug == 'gp_mean':
                x_quad, _, x_quad_test, y_quad_test, y_quad = data_aug_gp.quad_gp(test_fold=test_fold_num, **params)
            elif data_aug == 'gp_samp':
                x_quad, y_quad, x_quad_test, y_quad_test, _ = data_aug_gp.quad_gp(test_fold=test_fold_num, **params)

            kf_quad = KFold(n_splits=vali_split, shuffle=True, random_state=123)

            x_quad_train, x_quad_vali, y_quad_train, y_quad_vali = [], [], [], []

            for train_index, vali_index in kf_quad.split(x_quad):
                x_quad_train.append(x_quad[train_index])
                y_quad_train.append(y_quad[train_index])
                x_quad_vali.append(x_quad[vali_index])
                y_quad_vali.append(y_quad[vali_index])

            # Generate data for triplet protocol
            data_trip, y_trip, data_trip_test, y_trip_test = data_aug_gp.triplet_dw_gen(test_fold=test_fold_num,
                                                                                        n_samples=1)
            # Split the triplet data into training and validation
            kf_triplet = KFold(n_splits=vali_split, shuffle=True, random_state=456)
            data_trip_train, data_trip_vali, y_trip_train, y_trip_vali = [], [], [], []

            for train_index, vali_index in kf_triplet.split(data_trip):
                data_trip_train.append(data_trip.iloc[train_index])
                data_trip_vali.append(data_trip.iloc[vali_index])
                y_trip_train.append(y_trip[train_index])
                y_trip_vali.append(y_trip[vali_index])

            data_gen_train, data_gen_vali, data_gen_test, y_train, y_vali, y_test = [], [], [], [], [], []
            # Put time information into dataframe
            for i in range(vali_split):
                data_stdp_train = data_aug_gp.STDP_dw_gen(x_stdp_train[i])
                data_stdp_vali = data_aug_gp.STDP_dw_gen(x_stdp_vali[i])

                data_quad_train = data_aug_gp.quad_dw_gen(x_quad_train[i])
                data_quad_vali = data_aug_gp.quad_dw_gen(x_quad_vali[i])

                data_gen_train.append(pd.concat([data_stdp_train, data_trip_train[i], data_quad_train], axis=0))
                data_gen_vali.append(pd.concat([data_stdp_vali, data_trip_vali[i], data_quad_vali], axis=0))

                y_train.append(np.concatenate([y_stdp_train[i], y_trip_train[i], y_quad_train[i]]))
                y_vali.append(np.concatenate([y_stdp_vali[i], y_trip_vali[i], y_quad_vali[i]]))

            data_stdp_test = data_aug_gp.STDP_dw_gen(x_stdp_test.reshape(-1, 1))
            data_quad_test = data_aug_gp.quad_dw_gen(x_quad_test.reshape(-1, 1))
            data_gen_test.append(pd.concat([data_stdp_test, data_trip_test, data_quad_test], axis=0))
            y_test = np.concatenate([y_stdp_test, y_trip_test, y_quad_test])

        # elif data_aug == 'raw_samp':
        #     pass
        # else:
        #     return 'Wrong augmentation!!'

    return data_gen_train, data_gen_vali, data_gen_test, y_train, y_vali, y_test