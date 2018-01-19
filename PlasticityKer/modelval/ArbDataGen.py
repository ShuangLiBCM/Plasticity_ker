"""
    Functions for generating spike train from pairing ptl
"""
import numpy as np
import pdb

def arb_spk_gen(ptl, spk_reso, if_noise=0):
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
    spk_len = train_len//spk_reso * 1000     # spk_reso is in msec

    pre_spk = np.zeros(spk_len)
    post_spk = np.zeros(spk_len)

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

        else:
            print('Wrong time interval!')

    # pre-post: 5-5
    elif (int(ptl.pre_spk_num) == 5) & (int(ptl.post_spk_num) == 5):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        isi = int(1000 / ptl.pre_spk_freq / spk_reso)
        spk_time_base5 = np.hstack([spk_time_base + isi * i for i in range(5)])

        spk_time_pre = np.sort(np.concatenate([spk_time_base5 + rep_interval * i for i in range(rep_num)]))
        mean_dt = int(ptl.dt1/spk_reso)
        if if_noise & (mean_dt > 0):
            between_noise = np.random.normal(loc=0.0, scale=np.min([np.abs(mean_dt/2), 2]), size=spk_time_pre.shape).astype(int)
            between_noise = between_noise * (np.abs(between_noise) < np.abs(mean_dt))
            spk_time_post = spk_time_pre + mean_dt + between_noise
        else:
            spk_time_post = spk_time_pre + mean_dt

    else:
        print('Wrong ptl!')

    spk_time_pre[np.where(spk_time_pre > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    spk_time_post[np.where(spk_time_post > spk_len)] = spk_len - 1  # Prevent spike outsize protocol
    pre_spk[spk_time_pre] = 1
    post_spk[spk_time_post] = 1

    return spk_time_pre, spk_time_post, pre_spk, post_spk


def arb_w_gen(pre_spk, post_spk, kernel, network):
    """
    Generate arbitrary target w with given spike trains, kernels and network
    ------------------------------
    :param pre_spk: binary array indicating pre-synaptic train
    :param post_spk: binary array indicating post-synaptic train
    :param kernel: kernel object
    :param network: network object
    :return:
    """


