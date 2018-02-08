"""
Visualize spike train raster plots of a certain protocol
"""
import matplotlib.pyplot as plt
import numpy as np
import pdb

def spk_see(ptl_type=1, spk_pairs=None):

    if ptl_type == 1:
        # Obtain the pre-post scatter plot
        loci_pre = []
        loci_post = []
        dt_mean = []
        for i in range(spk_pairs.shape[0]):
            loci_pre_tmp = np.where(spk_pairs[i, :, 0] == 1)[0]
            loci_post_tmp = np.where(spk_pairs[i, :, 1] == 1)[0]
            loci_pre.append(loci_pre_tmp)
            loci_post.append(loci_post_tmp)
            dt_mean.append(np.mean(loci_pre_tmp - loci_post_tmp))

        sort_index = np.argsort(dt_mean)
        loci_pre_2 = [loci_pre[i] for i in sort_index]
        loci_post_2 = [loci_post[i] for i in sort_index]

    elif (ptl_type == 2) | (ptl_type == 4):
        loci_pre = []
        loci_post = []
        dt_mean = []
        for i in range(spk_pairs.shape[0]):
            loci_pre_tmp = np.where(spk_pairs[i, :, 0] == 1)[0]
            loci_post_tmp = np.where(spk_pairs[i, :, 1] == 1)[0]

            if len(loci_pre_tmp) == len(loci_post_tmp) * 2:  # Pre-post-pre
                loci_pre.append(loci_pre_tmp)
                loci_post.append(loci_post_tmp)
                index_pre = np.arange(0, len(loci_pre_tmp), 2)
                dt_mean.append(np.mean(loci_post_tmp - loci_pre_tmp[index_pre]))
            elif len(loci_post_tmp) == len(loci_pre_tmp) * 2:  # Post-pre-post
                loci_pre.append(loci_pre_tmp)
                loci_post.append(loci_post_tmp)
                index_post = np.arange(0, len(loci_post_tmp), 2)
                dt_mean.append(np.mean(loci_post_tmp[index_post] - loci_pre_tmp))
        
        sort_index = np.argsort(dt_mean)
        loci_pre_2 = [loci_pre[i] for i in sort_index]
        loci_post_2 = [loci_post[i] for i in sort_index]
        
    elif ptl_type == 3:
        pass
    else:
        pass

    return loci_pre_2, loci_post_2


def raster(event_times_list, trial_length,  **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial[trial_length]*2, ith + .5, ith + 1.5, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax