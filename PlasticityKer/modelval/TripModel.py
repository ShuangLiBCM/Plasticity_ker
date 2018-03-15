import numpy as np
from scipy.integrate import odeint
import pandas as pd
import tensorflow as tf
#from atflow.dataset import Dataset
#from atflow.trainers import Trainer
#from learning_plasticity.Kernelnet_triplet_shift import KernelNetTri_Shift
#from learning_plasticity import spk_gen_nonrand as spk_gen2
#from learning_plasticity import spk_gen
import pdb

# Define function for the dynamics of weights 
def fxn(s0,t,tau):
    # s0: initial condition for simulation
    # t: length of simulation
    # ix: extra argument
    
    ds = np.zeros(4)
    s = s0
 
    ds[0] = -(s[0])/tau[0]   # Simulating r1, A2+, LTP
    ds[1] = -(s[1])/tau[1]   # Simulating r2, A3-, trip-LTD
    ds[2] = -(s[2])/tau[2]   # Simulating o1, A2-, LTD
    ds[3] = -(s[3])/tau[3]   # Simulating o2, A3+, trip-LTP
    
    return ds

# Define All-to-All model
def trip_AlltoAll(a, tau, loci_track_pre, loci_track_post, ifSTDP=0, reso = 2, tt_len = 60, simu_step=1):
    # a: 4 *1 array, float, amplitude parameter
    # tau: 4 * 1 array, float, time constant
    # loci_track_pre: n * m, n is the sample size, m is number of pre spikes
    # loci_track_post: n * m, n is the sample size, m is the numebr of post spikes

    S_all = []
    w_all = []
    dw_final = np.zeros(len(loci_track_pre))
    for j in range(len(loci_track_pre)):
        pre_spk = loci_track_pre[j]
        post_spk = loci_track_post[j]

        spk_idx_tmp = np.sort(np.concatenate((pre_spk, post_spk)))
        spk_idx = np.zeros(spk_idx_tmp.shape[0]+2)        # total spike len considering sampling resolution
        spk_idx[1:spk_idx_tmp.shape[0]+1] = spk_idx_tmp
        spk_idx[spk_idx.shape[0]-1] = int(np.ceil(tt_len/reso*1000))
        spk_idx = np.unique(spk_idx.astype(int))

        s0 = np.zeros(4)   
        S_track = []
        w0 = 0
        w = [np.zeros(int(spk_idx[1]))]

        for i in range(spk_idx.shape[0]-1):
            t_simu = np.arange(spk_idx[i], spk_idx[i+1]+1, simu_step)
            wt = np.ones(len(t_simu)) * w0
                
            dw = 0

            if (spk_idx[i] in pre_spk) & (spk_idx[i] in post_spk):
                s0 = s0 + np.array([1, 1, 1, 1])
                if ifSTDP:
                    dw = s0[0] * a[0] - s0[2] * a[2]   # pair term only
                    # dw = -s0[2]   # pair term only 
                else:
                    dw = s0[0] * (a[0] + a[3] * s[3])-s0[2] * (a[2] + a[1] * s[1])
            elif spk_idx[i] in pre_spk:
                s0 = s0 + np.array([1, 1, 0, 0])
                # s0 = s0 + np.array([a[0],a[1],0,0])
                if ifSTDP:
                    dw = -s0[2] * a[2]   # pair term only
                    # dw = -s0[2]   # pair term only 
                else:
                    dw = -s0[2] * (a[2] + a[1] * s[1])
                    # dw = -s0[2] * (1 + s[1])
            elif spk_idx[i] in post_spk:
                # s0 = s0 + np.array([0,0,a[2],a[3]])
                s0 = s0 + np.array([0, 0, 1, 1])
                if ifSTDP:
                    dw = s0[0] * a[0]    # pair term only
                   # dw = s0[0]
                else:
                    dw = s0[0] * a[0] + s[0] * a[3] * s[3]
                   # dw = s0[0] * (1 + s[3])

            wt = wt + dw
            S = odeint(fxn, s0, t_simu, args=(tau,))
            s0 = S[S.shape[0]-1]
            s = S[S.shape[0]-2]
            S_track.append(S[:S.shape[0]-1])
            w.append(wt)
            w0 = wt[len(wt)-1]
            
        # 60 s of ds and w for one datapoint
        S_final = np.vstack(S_track)
        w_final = np.hstack(w)
        dw_final[j] = w_final[-1]
        w_all.append(w_final)
        S_all.append(S_final)
    
    return w_all, S_all, dw_final

# Define Nearest-spk model
def trip_NearestSpk(a, tau, loci_track_pre, loci_track_post,  ifSTDP=0, reso = 2, tt_len = 60, simu_step=1):
    # a: 4 *1 array, float, amplitude parameter
    # tau: 4 * 1 array, float, time constant
    # pre_loci: n * m, n is the sample size, m is number of pre spikes
    # post_loci: n * m, n is the sample size, m is the numebr of post spikes

    S_all = []
    w_all = []
    dw_final = np.zeros(len(loci_track_pre))
    for j in range(len(loci_track_pre)):
        pre_spk = loci_track_pre[j]
        post_spk = loci_track_post[j]

        spk_idx_tmp = np.sort(np.concatenate((pre_spk,post_spk)))
        spk_idx = np.zeros(spk_idx_tmp.shape[0]+2)        # total spike len considering sampling resolution
        spk_idx[1:spk_idx_tmp.shape[0]+1]=spk_idx_tmp
        spk_idx[spk_idx.shape[0]-1] = int(np.ceil(tt_len/reso*1000))
        spk_idx = np.unique(spk_idx.astype(int))

        s0 = np.zeros(4)   
        S_track = []
        w = []
        w0 = 0

        for i in range(spk_idx.shape[0]-1):
            simu_num = spk_idx[i+1] - spk_idx[i]
            t_simu = np.arange(spk_idx[i], spk_idx[i+1]+1,simu_step)
            ix = np.zeros(2)
            wt = np.ones(len(t_simu)) * w0
                
            dw = 0
            if spk_idx[i] in pre_spk:
                s = s0
                s0 = np.array([1,1,s0[2],s0[3]])
                if ifSTDP:
                    dw = -s0[2] * a[2]   # pair term only 
                else:
                    dw = -s0[2] * (a[2] + a[1] * s[1])

            if spk_idx[i] in post_spk:
                s = s0
                s0 = np.array([s0[0],s0[1],1,1])
                if ifSTDP:
                    dw = s0[0] * a[0]    # pair term only 
                else:
                    dw = s0[0] * a[0] + s[0] * a[3] * s[3]
                
            wt = wt + dw
            S = odeint(fxn,s0,t_simu, args=(tau,))
            s0 = S[S.shape[0]-1]
            S_track.append(S[:S.shape[0]-1])
            w.append(wt)
            w0 = wt[len(wt)-1]

        # 60 s of ds and w for one datapoint
        S_final = np.vstack(S_track)
        w_final = np.hstack(w)
        dw_final[j] = w_final[-1]
        w_all.append(w_final)
        S_all.append(S_final)
    
    return w_all, S_all, dw_final