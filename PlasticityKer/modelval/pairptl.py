"""
    Define the PairPtl class
"""

class PairPtl(object):

    def __init__(self, ptl_idx, pre_spk_num, pre_spk_freq, post_spk_num, post_spk_freq,
                 ptl_occ, ptl_freq, dt1, dt2, dt3, dw_mean, dw_ste, verbose=0):
        """
        Convert data frame entry into object attributes
        :param ptl_idx:
        :param pre_spk_num:
        :param pre_spk_freq:
        :param post_spk_num:
        :param post_spk_freq:
        :param ptl_occ:
        :param ptl_freq:
        :param dt1:
        :param dt2:
        :param dt3:
        :param dw_mean:
        :param dw_ste:
        :param verbose:
        """
        self.ptl_idx = ptl_idx
        self.pre_spk_num = pre_spk_num
        self.pre_spk_freq = pre_spk_freq
        self.post_spk_num = post_spk_num
        self.post_spk_freq = post_spk_freq
        self.ptl_occ = ptl_occ
        self.ptl_freq = ptl_freq
        self.dt1 = dt1
        self.dt2 = dt2
        self.dt3 = dt3
        self.dw_mean = dw_mean
        self.dw_ste = dw_ste

        if verbose == 1:
            if self.ptl_idx == 1:
                print('Protocol Bi&Poo, 1998, Fig7 (STDP), pre_spk_num=%d, post_spk_num=%.2f,dt =%.2f' % (pre_spk_num, post_spk_num, dt1))
            elif self.ptl_idx == 2:
                print('Wang&Bi, 2005, Fig7 (Triplet), pre_spk_num=%d, post_spk_num=%d, dt1=%.2f, dt2=%.2f' % (pre_spk_num, post_spk_num, dt1, dt2))
            elif self.ptl_idx == 3:
                print('Wang&Bi, 2005, Fig7 (Quadruplet), pre_spk_num=%d, post_spk_num=%d, dt1=%.2f, dt2=%.2f, dt3=%.2f' % (pre_spk_num, post_spk_num, dt1, dt2, dt3))
            elif self.ptl_idx == 4:
                print('Wang&Bi, 2005, Fig6C (Triplet2), pre_spk_num=%d, post_spk_num=%d, dt1=%.2f, dt2=%.2f' % (pre_spk_num, post_spk_num, dt1, dt2))
            elif self.ptl_idx == 5:
                print('Sjostrom&Nelson, 2001, Fig1D (Freq dependent LTP), pre_spk_num=%d, post_spk_num=%d,dt=%.2f' % (pre_spk_num, post_spk_num, dt1))
            elif self.ptl_idx == 6:
                print('Sjostrom&Nelson, 2001, Fig7B (Freq dependent LTD), pre_spk_num=%d, post_spk_num=%d,dt=%.2f' % (pre_spk_num, post_spk_num, dt1))
            elif self.ptl_idx == 7:
                print('Sjostrom&Nelson, 2001, Fig7C (STDP, 0.1Hz, 20Hz, 50Hz), pre_spk_num=%d, post_spk_num=%.d,dt=%.2f' % (pre_spk_num, post_spk_num, dt1))
            elif self.ptl_idx == 8:
                print('Sjostrom&Nelson, 2001, Fig7D (Freq dependence, 0ms), pre_spk_num=%d, post_spk_num=%d,dt=%.2f' % (pre_spk_num, post_spk_num, dt1))


