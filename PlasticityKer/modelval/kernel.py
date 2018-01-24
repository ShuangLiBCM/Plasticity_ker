"""
    define kernel class
"""
import numpy as np

class KernelGen(object):

    def __init__(self, tau_left=100, tau_right=20, scale_left=1, scale_right=1, scale=1, side='right', tau=20,
                 reso_kernel=2, len_kernel=51, kernel_pre_post=None, kernel_post_pre=None, kernel_post_post=None):
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
        self.scale_left = scale_left
        self.scale_right = scale_right
        self.scale = scale
        self.side = side
        self.bilat_ker = self.bi_exp_ker()
        self.unilat_ker = self.uni_exp_ker()
        self.dot_ker = self.dot_ker()

        if kernel_pre_post is not None:
            self.kernel_pre_post = kernel_pre_post
            self.kernel_post_pre = kernel_post_pre
            self.kernel_post_post = kernel_post_post

    def bi_exp_ker(self):
        """
            Implement bilateral exponential decay kernel
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1)/2)
        left_x = self.x[mid_pt] - self.x[0:mid_pt]
        right_x = self.x[mid_pt:]
        kernel[0:mid_pt] = -1 * np.exp(-1 * left_x / self.tau_left) * self.scale_left
        kernel[mid_pt:] = np.exp(-1 * right_x / self.tau_right) * self.scale_right
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def uni_exp_ker(self, side=None, tau=None, scale=None):
        """
            Implement unilateral exponential decay kernel
        """
        if side is None:
            side = self.side

        if tau is None:
            tau = self.tau

        if scale is None:
            scale = self.scale

        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1) / 2)
        if side == 'right':
            right_x = self.x[mid_pt:]
            kernel[mid_pt:] = np.exp(-1 * right_x / tau) * scale
        else:
            left_x = self.x[mid_pt] - self.x[0:mid_pt]
            kernel[0:mid_pt] = np.exp(-1 * left_x / tau) * scale
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def dot_ker(self):
        """
            Implement kernel that preserves original data
        """
        kernel = np.zeros(self.len_kernel)
        mid_pt = int((self.len_kernel - 1) / 2)
        kernel[mid_pt] = 1 * self.scale
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel