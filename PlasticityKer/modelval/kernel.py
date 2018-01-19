"""
    define kernel class
"""
import numpy as np

class KernelGen(object):

    def __init__(self, reso_kernel=2, len_kernel=101):
        """
        input
        reso: int, in ms, resolution of kernel
        len: odd int, length of the kernel
        """
        self.reso_kernel = reso_kernel
        self.len_kernel = len_kernel
        self.x = np.linspace(-1 * (self.len_kernel-1)/2, (self.len_kernel-1)/2 + 1, self.len_kernel) * self.reso_kernel
        self.kernel = np.zeros(self.len_kernel)

    def bi_exp_ker(self, tau_left=20, tau_right=20):
        """
            Implement bilateral exponential decay kernel
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1)/2)
        left_x = self.x[mid_pt] - self.x[0:mid_pt]
        right_x = self.x[mid_pt:]
        kernel[0:mid_pt] = -1 * np.exp(-1 * left_x / tau_left)
        kernel[mid_pt:] = np.exp(-1 * right_x / tau_right)
        self.kernel = kernel

        return kernel

    def uni_exp_ker(self, tau=20, side='right'):
        """
            Implement unilateral exponential decay kernel
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1) / 2)
        if side == 'right':
            right_x = self.x[mid_pt:]
            kernel[mid_pt:] = np.exp(-1 * right_x / tau)
        else:
            left_x = self.x[mid_pt] - self.x[0:mid_pt]
            kernel[0:mid_pt] = np.exp(-1 * left_x / tau)
        self.kernel = kernel

        return kernel