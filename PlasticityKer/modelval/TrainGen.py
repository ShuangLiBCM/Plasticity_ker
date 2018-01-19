"""
    Functions for generating spike train from pairing protocol
"""
import numpy as np

def arb_spk_w_gen(protocol, kernel, noise=[0,0,0]):
    """
    Generate a single set of pre-spike, post-spike trains and target given the protocol and ground truth kernel
    -------------------------------
    input:
    protocol: PairPtl object
    kernel: Ground truth kernel from Kernel object
    noise: list of booleans, introducing noise for within pre-spike, between pre-post, within post
    spk_train_len: int, length of the spike train, in sec

    output:
    X_pre: 2d array, pre-synaptic spike trains
    X_post: 2d array, post-synaptic spike train
    y: 1d array, target weights change
    """

    # Define resolution of spike train
    train_len = int(protocol.ptl_occ / protocol.ptl_freq)
    spk_reso = kernel.reso_kernel

    # Define length of the spike train
    spk_len = train_len//spk_reso * 1000     # spk_reso is in msec

    pre_spk = np.zeros(spk_len)
    post_spk = np.zeros(spk_len)

    rep_interval = int(np.floor(1000 / protocol.ptl_freq / spk_reso))
    rep_num = int(np.floor(train_len * protocol.ptl_freq))

    # Consider the max dt = 120 ms
    min_bef = int(120//spk_reso + 1)

    # pre-post: 1-1
    if (int(protocol.pre_spk_num) == 1) & (int(protocol.post_spk_num) == 1):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        # Obtain time index of spike
        spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
        spk_time_post = spk_time_pre + int(protocol.dt1 / spk_reso)    # dt1, dt2, dt3 in ms
        # Obtain spike train

    # pre-post: 2-1
    elif (int(protocol.pre_spk_num) == 2) & (int(protocol.post_spk_num) == 1):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        # Obtain time index of spike
        spk_time_post = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
        spk_time_pre1 = spk_time_post + int(protocol.dt1 / spk_reso)
        spk_time_pre2 = spk_time_post + int(protocol.dt2 / spk_reso)
        spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))

    # pre-post: 1-2
    elif (int(protocol.pre_spk_num) == 1) & (int(protocol.post_spk_num) == 2):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        # Obtain time index of spike
        spk_time_pre = np.hstack([spk_time_base + rep_interval * i for i in range(rep_num)])
        spk_time_post1 = spk_time_pre - int(protocol.dt1 / spk_reso)
        spk_time_post2 = spk_time_pre - int(protocol.dt2 / spk_reso)
        spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

    # pre-post: 2-2
    elif (int(protocol.pre_spk_num) == 2) & (int(protocol.post_spk_num) == 2):
        spk_time_base1 = np.random.randint(min_bef, rep_interval - min_bef, 1)
        spk_time_base2 = spk_time_base1 + int(np.abs(protocol.dt2) / spk_reso)

        if protocol.dt2 < 0:  # Pre-post-post-pre
            spk_time_post1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_post2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])
            spk_time_pre1 = spk_time_post1 - int(protocol.dt1 / spk_reso)
            spk_time_pre2 = spk_time_post2 - int(protocol.dt3 / spk_reso)
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))


        elif protocol.dt2 > 0:   # Post-pre-pre-post
            spk_time_pre1 = np.hstack([spk_time_base1 + rep_interval * i for i in range(rep_num)])
            spk_time_pre2 = np.hstack([spk_time_base2 + rep_interval * i for i in range(rep_num)])
            spk_time_post1 = spk_time_pre1 - int(protocol.dt1 / spk_reso)
            spk_time_post2 = spk_time_pre2 - int(protocol.dt3 / spk_reso)
            # Obtain spike train
            spk_time_pre = np.sort(np.concatenate([spk_time_pre1, spk_time_pre2]))
            spk_time_post = np.sort(np.concatenate([spk_time_post1, spk_time_post2]))

        else:
            print('Wrong time interval!')

    # pre-post: 5-5
    elif (int(protocol.pre_spk_num) == 5) & (int(protocol.post_spk_num) == 5):
        spk_time_base = np.random.randint(min_bef, rep_interval - min_bef, 1)
        isi = int(1000 / protocol.pre_spk_freq / spk_reso)
        spk_time_base5 = np.hstack([spk_time_base + isi * i for i in range(5)])

        spk_time_pre = np.sort(np.concatenate([spk_time_base5 + rep_interval * i for i in range(rep_num)]))
        spk_time_post = spk_time_pre + int(protocol.dt1 / spk_reso)

    else:
        print('Wrong protocol!')

    pre_spk[spk_time_pre] = 1
    post_spk[spk_time_post] = 1

    return spk_time_pre, spk_time_post, pre_spk, post_spk




