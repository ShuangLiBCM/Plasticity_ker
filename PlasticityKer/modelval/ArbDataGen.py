"""
    Functions for generating spike train from pairing ptl
"""
import numpy as np
from modelval import pairptl, network, trainer

def arb_spk_gen(ptl, spk_reso, spk_len=None, if_noise=1):
    """
    Generate a single set of pre-spike, post-spike trains and target given the ptl
    -------------------------------
    input:
    ptl: PairPtl object
    spk_reso: Sampling resolution in ms, should be consistent with kernel sampling resolution
    if_noise: list of booleans, enabling noise or not for within spike, between pre-post
    spk_train_len: int, length of the spike train, in sec

    output:
    spk_time_pre: location of pre-synaptic spike
    spk_time_post: location of post-synaptic spike
    pre_spk: pre-synaptic spike train
    post_spk: post-synaptic spike train
    """

    # Define resolution of spike train
    train_len = int(ptl.ptl_occ / ptl.ptl_freq)

    # Define length of the spike train
    if spk_len is None:
        spk_len = train_len//spk_reso * 1000     # spk_reso is in msec

    spk_pair = np.zeros((spk_len,2))

    rep_interval = int(np.floor(1000 / ptl.ptl_freq / spk_reso))
    rep_num = int(np.floor(train_len * ptl.ptl_freq))

    # Consider the max dt = 120 ms
    min_bef = int(120//spk_reso + 1)

    # pre-post: 1-1
    if (int(ptl.pre_spk_num) == 1) & (int(ptl.post_spk_num) == 1):

        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        # Obtain time index of spike
        spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
        mean_dt = int(ptl.dt1 / spk_reso)

        # Generate noise
        if if_noise:
            between_noise = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt/2), 2]), size=spk_time_pre.shape).astype(int)
            between_noise = between_noise * (np.abs(between_noise) < np.abs(mean_dt))
            spk_time_post = spk_time_pre + mean_dt + between_noise   # dt1, dt2, dt3 in ms
        else:
            spk_time_post = spk_time_pre + mean_dt # dt1, dt2, dt3 in ms

    # pre-post: 2-1
    elif (int(ptl.pre_spk_num) == 2) & (int(ptl.post_spk_num) == 1):

        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        mean_dt1 = int(ptl.dt1 / spk_reso)
        mean_dt2 = int(ptl.dt2 / spk_reso)
        # Obtain time index of spike
        if if_noise:
            spk_time_post = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            between_noise1 = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt1/2), 2]), size=spk_time_post.shape).astype(int)
            between_noise1 = between_noise1 * (np.abs(between_noise1) < np.abs(mean_dt1))
            spk_time_pre1 = spk_time_post + mean_dt1 + between_noise1
            between_noise2 = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt2/2), 2]), size=spk_time_post.shape).astype(int)
            between_noise2 = between_noise2 * (np.abs(between_noise2) < np.abs(mean_dt2))
            spk_time_pre2 = spk_time_post + mean_dt2 + between_noise2
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
        else:
            spk_time_post = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
            spk_time_pre1 = spk_time_post + mean_dt2
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
            between_noise1 = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt1/2), 2]), size=spk_time_pre.shape).astype(int)
            between_noise1 = between_noise1 * (np.abs(between_noise1) < np.abs(ptl.dt1/spk_reso))
            spk_time_post1 = spk_time_pre - mean_dt1 + between_noise1
            between_noise2 = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt2/2), 2]), size=spk_time_pre.shape).astype(int)
            between_noise2 = between_noise2 * (np.abs(between_noise2) < np.abs(mean_dt2))
            spk_time_post2 = spk_time_pre - mean_dt2 + between_noise2
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
        if if_noise:
            within_noise = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt/2), 2]), size=spk_time_base1.shape).astype(int)
            within_noise = within_noise * (np.abs(within_noise) < np.abs(mean_dt))
            spk_time_base2 = spk_time_base1 + np.abs(mean_dt) + within_noise
        else:
            spk_time_base2 = spk_time_base1 + np.abs(mean_dt)

        if ptl.dt2 < 0:  # Pre-post-post-pre
            spk_time_post1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_post2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])
            between_noise1 = np.random.normal(loc=0.0, scale=np.min([np.abs(ptl.dt1 / spk_reso), 2]),
                                              size=spk_time_post1.shape).astype(int)
            between_noise1 = between_noise1 * (np.abs(between_noise1) < np.abs(ptl.dt1/spk_reso))
            between_noise2 = np.random.normal(loc=0.0, scale=np.min([np.abs(ptl.dt3 / spk_reso), 2]),
                                              size=spk_time_post2.shape).astype(int)
            between_noise2 = between_noise2 * (np.abs(between_noise2) < np.abs(ptl.dt3/spk_reso))
            spk_time_pre1 = spk_time_post1 - int(ptl.dt1 / spk_reso) + between_noise1
            spk_time_pre2 = spk_time_post2 - int(ptl.dt3 / spk_reso) + between_noise2
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

        elif ptl.dt2 > 0:   # Post-pre-pre-post
            spk_time_pre1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_pre2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])
            between_noise1 = np.random.normal(loc=0.0, scale=np.min([np.abs(ptl.dt1 / spk_reso), 2]),
                                              size=spk_time_pre1.shape).astype(int)
            between_noise1 = between_noise1 * (np.abs(between_noise1) < np.abs(ptl.dt1/spk_reso))
            between_noise2 = np.random.normal(loc=0.0, scale=np.min([np.abs(ptl.dt3 / spk_reso), 2]),
                                              size=spk_time_pre2.shape).astype(int)
            between_noise2 = between_noise2 * (np.abs(between_noise2) < np.abs(ptl.dt3/spk_reso))
            spk_time_post1 = spk_time_pre1 - int(ptl.dt1 / spk_reso) + between_noise1
            spk_time_post2 = spk_time_pre2 - int(ptl.dt3 / spk_reso) + between_noise2
            # Obtain spike train
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

        else:
            print('Wrong time interval!')

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
                        if spk_time_post[i] >= spk_pair.shape[0]:
                            spk_time_post[i] = spk_pair.shape[0]-1
                    elif -mean_dt - between_noise[i] >= isi:
                        spk_time_post[i] = int(spk_time_pre[i] - isi + 1)
                    else:
                        spk_time_post[i] = int(spk_time_pre[i] + mean_dt + between_noise[i])
                else:
                    if mean_dt + between_noise[i] < 0:
                        spk_time_post[i] = int(spk_time_pre[i] + 1)
                        if spk_time_post[i] >= spk_pair.shape[0]:
                            spk_time_post[i] = spk_pair.shape[0]-1
                    elif mean_dt + between_noise[i] >= isi:
                        spk_time_post[i] = int(spk_time_pre[i] + isi)
                    else:
                        spk_time_post[i] = int(spk_time_pre[i] + mean_dt + between_noise[i])
        else:
            spk_time_post = spk_time_pre + mean_dt
        
    else:
        print('Wrong ptl!')

    spk_time_pre[np.where(spk_time_pre > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    spk_time_post[np.where(spk_time_post > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    spk_pair[spk_time_pre, 0] = 1
    spk_pair[spk_time_post, 1] = 1

    return spk_time_pre, spk_time_post, spk_pair


def arb_w_gen(spk_pairs=None, df=None, ptl_list=None, kernel=None, spk_len=None, aug_times=None, net_type='pair'):
    """
    Generate arbitrary target w with given spike trains, kernel and network
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

        if spk_len is None:
            spk_len = int(15 / 0.1 * 1000 / kernel.reso_kernel)   # The longest protocol

        for i in range(len(ptl_list)):
            data_ptl = df[df['ptl_idx'] == ptl_list[i]]

            for j in range(len(data_ptl)):
                ptl_info = pairptl.PairPtl(*data_ptl.iloc[j])
                for _ in range(aug_times[i]):
                    _, _, spk_pair = arb_spk_gen(ptl_info, kernel.reso_kernel, spk_len=spk_len, if_noise=1)
                    spk_pairs.append(spk_pair)

        # Generate the spike data
        spk_pairs = np.array(spk_pairs)   # Check the dimension into  (m * n * 2)
    
    # Get the network used to generate prediction
    if net_type == 'pair':
        gen_pairnet = network.PairNet(kernel=kernel, n_input=spk_pairs.shape[1])
        # Send the network graph into trainer, and name of placeholder
        gen_pairnet_train = trainer.Trainer(gen_pairnet.prediction, gen_pairnet.prediction, input_name=gen_pairnet.inputs)

        # generate targets through evaluating the prediction of the network
        targets = gen_pairnet_train.evaluate(ops=gen_pairnet.prediction, inputs=spk_pairs)

    elif net_type == 'triplet':
        gen_tripnet = network.TripNet(kernel=kernel, n_input=spk_pairs.shape[1])

        # Send the network graph into trainer, and name of placeholder
        gen_tripnet_train = trainer.Trainer(gen_tripnet.prediction, gen_tripnet.prediction,
                                            input_name=gen_tripnet.inputs)

        # generate targets through evaluating the prediction of the network
        targets = gen_tripnet_train.evaluate(ops=gen_tripnet.prediction, inputs=spk_pairs)
    else:
        print('Wrong network!!')


    return spk_pairs, targets