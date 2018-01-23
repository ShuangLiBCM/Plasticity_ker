"""
    define kernel class
"""
import numpy as np

class KernelGen(object):

    def __init__(self, tau_left=10, tau_right=10, tau=10, side='right', reso_kernel=2, len_kernel=51):
        """
        Created and build the kernel
        :param tau_left: exponential decay tau for left side of kernel
        :param tau_right: exponential decay tau for right side of kernel
        :param tau: exponential decay tau for one side kernel
        :param side: 'left' or 'right', for choosing side of the uni-side kernel
        :param reso_kernel: sampling resolution of kernel in ms, 2 ms by default
        :param len_kernel: length of kernel, should be an odd number, 101 by default
        """
        self.reso_kernel = reso_kernel
        self.len_kernel = len_kernel
        self.x = np.linspace(-1 * (self.len_kernel-1)/2, (self.len_kernel-1)/2 + 1, self.len_kernel) * self.reso_kernel
        self.kernel = np.zeros(self.len_kernel)
        self.tau_left = tau_left
        self.tau_right = tau_right
        self.tau = tau
        self.side = side
        self.bilat_ker = self.bi_exp_ker()
        self.unilat_ker = self.uni_exp_ker()
        self.dot_ker = self.dot_ker()

    def bi_exp_ker(self):
        """
            Implement bilateral exponential decay kernel
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1)/2)
        left_x = self.x[mid_pt] - self.x[0:mid_pt]
        right_x = self.x[mid_pt:]
        kernel[0:mid_pt] = -1 * np.exp(-1 * left_x / self.tau_left)
        kernel[mid_pt:] = np.exp(-1 * right_x / self.tau_right)
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def uni_exp_ker(self):
        """
            Implement unilateral exponential decay kernel
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1) / 2)
        if self.side == 'right':
            right_x = self.x[mid_pt:]
            kernel[mid_pt:] = np.exp(-1 * right_x / self.tau)
        else:
            left_x = self.x[mid_pt] - self.x[0:mid_pt]
            kernel[0:mid_pt] = np.exp(-1 * left_x / self.tau)
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def dot_ker(self):
        """
            Implement kernel that preserves original data
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1) / 2)
        kernel[mid_pt] = 1
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel