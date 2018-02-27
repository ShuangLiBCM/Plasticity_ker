# This code contains functions that take the mean dt time, required length & randomness, and generate pre and post spike trains
import numpy as np
import pdb

def get_stdp_spikes(dt=0, rep=10, pre_rand=None, post_rand=None, ptl_occ=60, reso = 2, freq = 1, target = None, dt_add=None, within_train_jitter=5, between_train_jitter=0.1):
    # dt: float or float arrays, input time difference in ms
    # rep: int, number of augmentations per dt value
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, Gaussian by default
    # ptl_occ: int, number of occurrence of the protocol
    # reso: float, resolution per sample points in ms
    # freq: float, frequency of the protocol repetition in Hz
    # dt_add: list of float, added spike time differences there are not on the original protocol
    # within_train_jitter: int, control variance of the pre-spike time
    # betwee_train_jitter: int, control variance of the pre-post time differences

    data_len = int(freq * ptl_occ * 1000 / reso)
    pre_spk = np.zeros(shape = (len(dt) * rep, data_len))   # Initialize pre- and post- spike trains
    post_spk = np.zeros(shape = (len(dt) * rep, data_len))
    
    target_new = []
    if target is not None:
        target_new = np.zeros(shape = (len(dt) * rep, 1))

    len_ptl = 1000/ reso / freq    # Inter-protocol difference after considering sampling resolution

    k = 0
    for i in range(len(dt)):
        dt_tmp = dt[i]
        for j in (np.arange(rep)):
            # Generate pre_spk_train
            # Choose where to initiate the spk
            np.random.seed(None)
            mid_ptl = np.random.uniform(low=150, high=len_ptl - 150)   # Maximum dt length is 120 ms

            # Jitter to be put on each following spikes
            jitter_pre = np.random.normal(loc=0.0, scale=within_train_jitter, size = ptl_occ)

            # Generate post_spk_train
            jitter_post = np.random.normal(loc=0.0, scale=between_train_jitter, size = ptl_occ)

            pre_one_idx = np.asarray([mid_ptl + m * len_ptl + jitter_pre[m] for m in range(ptl_occ)])
            post_one_idx = np.asarray([np.clip(pre_one_idx[n] + jitter_post[n] +  dt_tmp/reso, len_ptl * n, (n+1) * len_ptl -1) for n in range(ptl_occ)])

            pre_spk[k, pre_one_idx.astype(int)] = 1
            post_spk[k, post_one_idx.astype(int)] = 1

            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1

    return pre_spk, post_spk, target_new


def trip_spk(dt1=0, dt2=0, rep=10, pre_rand=None, post_rand=None, ptl_occ=60, reso=2, freq=1, target=None, dt_add=np.nan, within_train_jitter=5, between_train_jitter=0.5):
    # dt: float or float arrays, input time difference in ms
    # rep: int, number of realizations per dt value
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, exp by default
    # ptl_occ: int, number of occurrence of the protocol
    # reso: float, resolution per sample points in ms
    # freq: float, frequency of the protocol repetition in Hz
    # within_train_jitter: int, control variance of the pre-spike time
    # between_train_jitter: int, control variance of the pre-post time differences


    if dt_add is not np.nan:
        dt1 = dt1 + dt_add
        neg_dt_add = [-1 * dt_add[i] for i in range(len(dt_add))]
        dt2 = dt2 + neg_dt_add

    data_len = int(freq * ptl_occ * 1000 / reso)       # Length of the whole spike train
    pre_spk = np.zeros(shape = (len(dt1) * rep, data_len))
    post_spk = np.zeros(shape = (len(dt1) * rep, data_len))
    jitter_pre = np.zeros(ptl_occ)
    jitter_post = np.zeros(ptl_occ)
    jitter_pre1 = np.zeros(ptl_occ)
    jitter_pre2  = np.zeros(ptl_occ)
    jitter_post1 = np.zeros(ptl_occ)
    jitter_post2 = np.zeros(ptl_occ)
    
    target_new = []
    if target is not None:
        target_new = np.zeros(shape = (len(dt1) * rep, 1))

    len_ptl = 1000 / freq / reso        # Inter-protocol interval

    k = 0
     
    for i in range(len(dt1)):
        dt_tmp1 = dt1[i]
        dt_tmp2 = dt2[i]
        
        for j in (np.arange(rep)):
            # Generate pre_spk_train
            # Choose where to initiate the spk
            np.random.seed(None)
            mid_ptl = np.random.uniform(low=150, high=len_ptl - 150)   # Maximum dt length is 120 ms
            dist_half = np.round(dt_tmp1 / reso / 2)
            
            if dt_tmp1 < 0:    # Pre-post-pre protocol
                jitter_post = np.random.normal(loc=0.0, scale=within_train_jitter, size = ptl_occ)
                post_idx = np.asarray([mid_ptl + m * len_ptl + jitter_post[m] for m in range(ptl_occ)])
                jitter_pre1 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                pre_idx1 = np.asarray([np.clip(post_idx[n] - dt_tmp1/reso+jitter_pre1[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                jitter_pre2 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                pre_idx2 = np.asarray([np.clip(post_idx[n] - dt_tmp2/reso+jitter_pre2[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                pre_spk[k, pre_idx1.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1
                pre_spk[k, pre_idx2.astype(int)] = 1
            elif dt_tmp1 > 0:   # Post-pre-post protocol
                jitter_pre = np.random.normal(loc=0.0, scale=within_train_jitter, size = ptl_occ)
                pre_idx = np.asarray([mid_ptl + m * len_ptl + jitter_pre[m] for m in range(ptl_occ)])
                jitter_post1 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                post_idx1 = np.asarray([np.clip(pre_idx[n] +dt_tmp1/reso+ jitter_post1[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                jitter_post2 = np.random.normal(loc=0, scale=between_train_jitter, size=ptl_occ)
                post_idx2 = np.asarray([np.clip(pre_idx[n] +dt_tmp2/reso+ jitter_post2[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])                
                post_spk[k, post_idx1.astype(int)] = 1
                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx2.astype(int)] = 1
            
            pre_idx = []
            post_idx = []
            pre_idx1 = []
            pre_idx2 = []
            post_idx1 = []
            post_idx2 = []
            
            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1

    return pre_spk, post_spk, target_new

def quad_spk(dt=0, rep=10, pre_rand=None, post_rand=None, ptl_occ=60, reso = 2, freq = 1, target = None, within_train_jitter1=1, within_train_jitter2=1 ):
    # dt: float or float arrays, input time difference in ms
    # rep: int, number of realizations per dt value
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, exp by default
    # ptl_occ: int, number of occurrence of the protocol
    # reso: float, resolution per sample points in ms
    # freq: float, frequency of the protocol repetition in Hz
    # within_train_jitter1: int, variance for first middle spike
    # within_train_jitter2: int, variance for second middle spike

    data_len = int(freq * ptl_occ * 1000 / reso)
    pre_spk = np.zeros(shape = (len(dt) * rep, data_len))
    post_spk = np.zeros(shape = (len(dt) * rep, data_len))

    if target is not None:
        target_new = np.zeros(shape = (len(dt) * rep, 1))
    else:
        target_new = []

    len_ptl = 1000 / freq / reso
        
    k = 0
    for i in range(len(dt)):
        dt_tmp = dt[i]
        
        for j in (np.arange(rep)):
            # Generate pre_spk_train
            # Choose where to initiate the spk
            np.random.seed(None)
            mid_ptl = np.random.uniform(low=150, high=len_ptl - 150)   # Maximum dt length is 120 ms
            dist_half = np.round(dt_tmp / reso / 2)
            
            if dt_tmp > 0:  # post-pre-pre-post
                jitter_pre1 = np.random.normal(loc=0.0, scale=within_train_jitter1, size = ptl_occ)
                jitter_pre2 = np.random.normal(loc=0.0, scale=within_train_jitter2, size = ptl_occ)
                pre_one_idx1 = np.asarray([mid_ptl + m * len_ptl + jitter_pre1[m] for m in range(ptl_occ)])
                pre_one_idx2 = np.asarray([np.clip(pre_one_idx1[n] +dt_tmp/reso+ jitter_pre2[n], len_ptl * n, (n+1) * len_ptl -1) for n in range(ptl_occ)])
                post_one_idx1 = pre_one_idx1 - 5/reso
                post_one_idx2 = pre_one_idx2 + 5/reso
            elif dt_tmp < 0:   # Pre post protocol
                jitter_post1 = np.random.normal(loc=0.0, scale=5, size = ptl_occ)
                jitter_post2 = np.random.normal(loc=0.0, scale=1, size = ptl_occ)
                post_one_idx1 = np.asarray([mid_ptl + m * len_ptl + jitter_post1[m] for m in range(ptl_occ)])
                post_one_idx2 = np.asarray([np.clip(post_one_idx1[n] - dt_tmp/reso+ jitter_post2[n], len_ptl * n, (n+1) * len_ptl -1) for n in range(ptl_occ)])
                pre_one_idx1 = post_one_idx1 - 5/reso
                pre_one_idx2 = post_one_idx2 + 5/reso
                
            pre_spk[k, pre_one_idx1.astype(int)] = 1
            post_spk[k, post_one_idx1.astype(int)] = 1
            pre_spk[k, pre_one_idx2.astype(int)] = 1
            post_spk[k, post_one_idx2.astype(int)] = 1
            
            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1

            # pdb.set_trace()

    return pre_spk, post_spk, target_new

def trip_spk2(dt1=0, dt2=0, rep=10, pre_rand=None, post_rand=None, ptl_occ=60, reso=2, freq=1, target=None, dt_add=np.nan, within_train_jitter=5, between_train_jitter=0.5):
    # dt: float or float arrays, input time difference in ms
    # rep: int, number of realizations per dt value
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, Gaussian by default
    # ptl_occ: int, number of occurrence of the protocol
    # reso: float, resolution per sample points in ms
    # freq: float, frequency of the protocol repetition in Hz
    # within_train_jitter: int, control variance of the pre-spike time
    # between_train_jitter: int, control variance of the pre-post time differences

    if dt_add is not np.nan:
        dt1 = dt1 + dt_add
        neg_dt_add = [-1 * dt_add[i] for i in range(len(dt_add))]
        dt2 = dt2 + neg_dt_add

    data_len = int(freq * ptl_occ * 1000 / reso)       # Length of the whole spike train
    pre_spk = np.zeros(shape = (len(dt1) * rep, data_len))
    post_spk = np.zeros(shape = (len(dt1) * rep, data_len))
    jitter_pre = np.zeros(ptl_occ)
    jitter_post = np.zeros(ptl_occ)
    jitter_pre1 = np.zeros(ptl_occ)
    jitter_pre2  = np.zeros(ptl_occ)
    jitter_post1 = np.zeros(ptl_occ)
    jitter_post2 = np.zeros(ptl_occ)
    
    target_new = []
    if target is not None:
        target_new = np.zeros(shape = (len(dt1) * rep, 1))

    len_ptl = 1000 / freq / reso        # Inter-protocol interval

    k = 0
     
    for i in range(len(dt1)):
        dt_tmp1 = dt1[i]
        dt_tmp2 = dt2[i]
        
        for j in (np.arange(rep)):
            # Generate pre_spk_train
            # Choose where to initiate the spk
            np.random.seed(None)
            mid_ptl = np.random.uniform(low=150, high=len_ptl - 150)   # Maximum dt length is 120 ms
            dist_half = np.round(dt_tmp1 / reso / 2)
            
            if dt_tmp1 < 0:    # Pre-post-pre protocol
                jitter_post = np.random.normal(loc=0.0, scale=within_train_jitter, size = ptl_occ)
                post_idx = np.asarray([mid_ptl + m * len_ptl + jitter_post[m] for m in range(ptl_occ)])
                jitter_pre1 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                pre_idx1 = np.asarray([np.clip(post_idx[n] + dt_tmp1/reso+jitter_pre1[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                jitter_pre2 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                pre_idx2 = np.asarray([np.clip(post_idx[n] + dt_tmp2/reso+jitter_pre2[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                pre_spk[k, pre_idx1.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1
                pre_spk[k, pre_idx2.astype(int)] = 1
            elif dt_tmp1 > 0:   # Post-pre-post protocol
                jitter_pre = np.random.normal(loc=0.0, scale=within_train_jitter, size = ptl_occ)
                pre_idx = np.asarray([mid_ptl + m * len_ptl + jitter_pre[m] for m in range(ptl_occ)])
                jitter_post1 = np.random.normal(loc=0, scale=between_train_jitter, size = ptl_occ)
                post_idx1 = np.asarray([np.clip(pre_idx[n] - dt_tmp1/reso+ jitter_post1[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                jitter_post2 = np.random.normal(loc=0, scale=between_train_jitter, size=ptl_occ)
                post_idx2 = np.asarray([np.clip(pre_idx[n] - dt_tmp2/reso+ jitter_post2[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])                
                post_spk[k, post_idx1.astype(int)] = 1
                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx2.astype(int)] = 1
            
            pre_idx = []
            post_idx = []
            pre_idx1 = []
            pre_idx2 = []
            post_idx1 = []
            post_idx2 = []
            
            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1

    return pre_spk, post_spk, target_new

def freq_LTP(dt=0, rep=10, pre_rand=None, post_rand=None, ptl_occ_arr=15, reso=2, freq=0.1, spk_num_arr=5, spk_freq_arr=1, target=None, within_train_jitter1=0.5, within_train_jitter2=5, between_train_jitter=0.5):
    # dt: float, fixed pre-post time difference
    # rep: number of augmentation for the protocol
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, Gaussian by default
    # ptl_occ_arr: int array, repitition number of the pairing protocol
    # reso: int, sampling resolution, 2 ms by default
    # freq: int, frequency for protocol repititon
    # spk_freq_arr: int array, spike frequency within the spike train
    # spk_num_arr: int array, number of spikes within each frequency
    # target: array of float, ground truth weight change
    # within_train_jitter: int, control variance of the pre-spike time
    # between_train_jitter: int, control variance of the pre-post time differences
    data_len = int(50 / 0.1 * 1000 / reso)       # Length of the whole spike train
    pre_spk = np.zeros(shape = (len(dt) * rep, data_len))
    post_spk = np.zeros(shape = (len(dt) * rep, data_len))
    
    target_new = []
    if target is not None:
        target_new = np.zeros(shape = (len(dt) * rep, 1))

    len_ptl = 1000 / freq / reso        # Inter-protocol interval

    k = 0
    
    for i in range(len(dt)):
        dt_tmp = dt[i]       # Read the fixed pre-post time difference
        ptl_occ = int(ptl_occ_arr[i])
        spk_num = int(spk_num_arr[i])
        spk_freq = spk_freq_arr[i]
        mid_ptl = []
        for j in (np.arange(rep)):
            if spk_num <5:         # Low frequency protocol
                mid_ptl = np.random.uniform(low=10, high=len_ptl - 10)   # Maximum dt length is 120 ms
                jitter_pre = np.random.normal(loc=0.0, scale=within_train_jitter2, size = ptl_occ)
                pre_idx = np.asarray([mid_ptl + m * len_ptl + jitter_pre[m] for m in range(ptl_occ)])
                jitter_post = np.random.normal(loc=0.0, scale=between_train_jitter, size = ptl_occ)
                post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso+jitter_post[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                
                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1

            else: 
                mid_ptl = np.random.uniform(low=200, high=len_ptl - 200)   # Maximum dt length is 120 ms
                jitter_pre1 = np.random.normal(loc=0.0, scale=within_train_jitter1, size = 5)
                pre_5spk_ini = np.linspace(mid_ptl, mid_ptl + (spk_num - 1) * 1000 / spk_freq / reso, 5)+jitter_pre1
                jitter_pre2 = np.random.normal(loc=0.0, scale=within_train_jitter2, size = ptl_occ)
                pre_idx = np.asarray([pre_5spk_ini + m * len_ptl + jitter_pre2[m] for m in range(ptl_occ)])
                pre_idx = pre_idx.reshape(1,75)[0]
                jitter_post = np.random.normal(loc=0.0, scale=between_train_jitter, size = ptl_occ * spk_num)
                post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso, len_ptl * int(np.floor(n/5)), (int(np.floor(n/5))+1) * len_ptl+1) for n in range(ptl_occ * spk_num)])
                # post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso, len_ptl * j, (j+1) * len_ptl+1) for n,j in zip(range(75), range(ptl_occ))])

                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1
            
            pre_idx = []
            post_idx = []
            pre_idx1 = []
            pre_idx2 = []
            post_idx1 = []
            post_idx2 = []
            
            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1    
    
    return pre_spk, post_spk, target_new

def freq_STDP(dt=0, rep=10, pre_rand=None, post_rand=None, ptl_occ_arr=15, reso=2, freq=0.1, spk_num_arr=5, spk_freq_arr=1, target=None, within_train_jitter1=0.5, within_train_jitter2=5, between_train_jitter=0.5):
    # dt: float, fixed pre-post time difference
    # rep: number of augmentation for the protocol
    # pre_rand: Way of randomization of pre spike trains, Gaussian by default
    # post_rand: Way of randomization of post spike trains, Gaussian by default
    # ptl_occ_arr: int array, repitition number of the pairing protocol
    # reso: int, sampling resolution, 2 ms by default
    # freq: int, frequency for protocol repititon
    # spk_freq_arr: int array, spike frequency within the spike train
    # spk_num_arr: int array, number of spikes within each frequency
    # target: array of float, ground truth weight change
    # within_train_jitter: int, control variance of the pre-spike time
    # between_train_jitter: int, control variance of the pre-post time differences
    data_len = int(50 / 0.1 * 1000 / reso)       # Length of the whole spike train
    pre_spk = np.zeros(shape = (len(dt) * rep, data_len))
    post_spk = np.zeros(shape = (len(dt) * rep, data_len))
    
    target_new = []
    if target is not None:
        target_new = np.zeros(shape = (len(dt) * rep, 1))

    len_ptl = 1000 / freq / reso        # Inter-protocol interval

    k = 0
    
    for i in range(len(dt)):
        dt_tmp = dt[i]       # Read the fixed pre-post time difference
        ptl_occ = int(ptl_occ_arr[i])
        spk_num = int(spk_num_arr[i])
        spk_freq = spk_freq_arr[i]
        for j in (np.arange(rep)):
            if spk_num <5:         # Low frequency protocol
                mid_ptl = np.random.uniform(low=10, high=len_ptl - 10)   # Maximum dt length is 120 ms
                jitter_pre = np.random.normal(loc=0.0, scale=within_train_jitter2, size = ptl_occ)
                pre_idx = np.asarray([mid_ptl + m * len_ptl + jitter_pre[m] for m in range(ptl_occ)])
                jitter_post = np.random.normal(loc=0.0, scale=between_train_jitter, size = ptl_occ)
                post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso+jitter_post[n], len_ptl * n, (n+1) * len_ptl+1) for n in range(ptl_occ)])
                
                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1

            else: 
                mid_ptl = np.random.uniform(low=200, high=len_ptl - 200)   # Maximum dt length is 120 ms
                jitter_pre1 = np.random.normal(loc=0.0, scale=within_train_jitter1, size = 5)
                pre_5spk_ini = np.linspace(mid_ptl, mid_ptl + (spk_num - 1) * 1000 / spk_freq / reso, num = 5)+jitter_pre1
                jitter_pre2 = np.random.normal(loc=0.0, scale=within_train_jitter2, size = ptl_occ)
                pre_idx = np.asarray([pre_5spk_ini + m * len_ptl + jitter_pre2[m] for m in range(ptl_occ)])
                pre_idx = pre_idx.reshape(1,75)[0]
                jitter_post = np.random.normal(loc=0.0, scale=between_train_jitter, size = ptl_occ * spk_num)
                post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso, len_ptl * int(np.floor(n/5)), (int(np.floor(n/5))+1) * len_ptl+1) for n in range(ptl_occ * spk_num)])
                # post_idx = np.asarray([np.clip(pre_idx[n] + dt_tmp/reso, len_ptl * j, (j+1) * len_ptl+1) for n,j in zip(range(75), range(ptl_occ))])

                pre_spk[k, pre_idx.astype(int)] = 1
                post_spk[k, post_idx.astype(int)] = 1
            
            pre_idx = []
            post_idx = []
            pre_idx1 = []
            pre_idx2 = []
            post_idx1 = []
            post_idx2 = []
            
            if target is not None:
                target_new[k] = target[i]/100

            k = k + 1    
    
    return pre_spk, post_spk, target_new