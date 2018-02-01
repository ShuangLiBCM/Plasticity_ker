"""
    define kernel class
"""
import numpy as np

class KernelGen(object):

    def __init__(self, tau_left=100, tau_right=20, scale_left=1, scale_right=1, scale=1, side='right', tau=20,
                 reso_kernel=2, len_kernel=51, kernel_scale=None):
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
        self.kernel_pre = self.dot_ker()
        self.kernel_post = self.bi_exp_ker()
        self.kernel_post_post = self.uni_exp_ker()
        self.dot_ker = self.dot_ker()
        self.kernel_scale = kernel_scale
        
    def bi_exp_ker(self, len_kernel=None):
        """
            Implement bilateral exponential decay kernel
        """
        if len_kernel is None:
            len_kernel = self.len_kernel
            
        kernel = np.zeros(len_kernel)
        mid_pt = int((len_kernel - 1)/2)
        left_x = self.x[mid_pt] - self.x[0:mid_pt]
        right_x = self.x[mid_pt:]
        kernel[0:mid_pt] = -1 * np.exp(-1 * left_x / self.tau_left) * self.scale_left
        kernel[mid_pt:] = np.exp(-1 * right_x / self.tau_right) * self.scale_right
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def uni_exp_ker(self, side=None, tau=None, scale=None, shift=None, len_kernel=None):
        """
            Implement unilateral exponential decay kernel
        """
        if len_kernel is None:
            len_kernel = self.len_kernel
            
        if side is None:
            side = self.side

        if tau is None:
            tau = self.tau

        if scale is None:
            scale = self.scale

        kernel = np.zeros(len_kernel)
        mid_pt = int((len_kernel - 1) / 2)
        if side == 'right':
            if shift == 1:
                right_x = self.x[mid_pt+2:] - self.x[mid_pt+2]
                kernel[mid_pt:] = np.exp(-1 * right_x / tau) * scale
            elif shift == -1:
                right_x = self.x[mid_pt:] - self.x[mid_pt]
                kernel[mid_pt:] = np.exp(-1 * right_x / tau) * scale
            else:
                right_x = self.x[mid_pt+1:] - self.x[mid_pt+1]
                kernel[mid_pt+1:] = np.exp(-1 * right_x / tau) * scale
        else:
            if shift == 1:
                left_x = self.x[mid_pt-1] - self.x[0:mid_pt]
                kernel[0:mid_pt-1] = np.exp(-1 * left_x[1:] / tau) * scale
            elif shift == -1:
                left_x = self.x[mid_pt] - self.x[0:mid_pt+1]
                kernel[0:mid_pt+1] = np.exp(-1 * left_x / tau) * scale
            else:
                left_x = self.x[mid_pt-1] - self.x[0:mid_pt]
                kernel[0:mid_pt] = np.exp(-1 * left_x / tau) * scale
                
        self.kernel = kernel.reshape(-1, 1)
        
        return self.kernel

    def dot_ker(self, len_kernel=None):
        """
            Implement kernel that preserves original data
        """
        if len_kernel is None:
            len_kernel = self.len_kernel
            
        kernel = np.zeros(len_kernel)
        mid_pt = int((len_kernel - 1) / 2)
        kernel[mid_pt] = 1 * self.scale
        self.kernel = kernel.reshape(-1, 1)

        return self.kernel

    def trip_model_ker(self, para, data_name='Hippocampus'):
        """
        Generate pre, post and postpost kernel based on Gerstner's paper
        :param para: parameter of the model
        """
        a = para[:4].values
        tau = para[4:].values
        reso_set = 2
        tau_pre_post = tau[0] / reso_set  # ms
        tau_post_pre = tau[2] / reso_set  # ms
        tau_post_post = tau[3] / reso_set  # ms

        self.kernel_pre = self.uni_exp_ker(side='left', tau=tau_pre_post, scale=1, shift=-1)
        ker_pre_norm = np.linalg.norm(self.kernel_pre, ord=2)
        self.kernel_pre = self.kernel_pre / ker_pre_norm
        self.kernel_post = self.uni_exp_ker(side='left', tau=tau_post_pre, scale=1)
        ker_post_norm = np.linalg.norm(self.kernel_post, ord=2)
        self.kernel_post = self.kernel_post / ker_post_norm
        self.kernel_post_post = self.uni_exp_ker(side='left', tau=tau_post_post, scale=1, shift=-1)
        ker_post_post_norm = np.linalg.norm(self.kernel_post_post, ord=2)
        self.kernel_post_post = self.kernel_post_post / ker_post_post_norm
        
        if data_name == 'Hippocampus':
            self.kernel_scale = np.array([a[0] * ker_pre_norm, a[2] * ker_post_norm, a[3]/a[0] * ker_post_post_norm])
        elif data_name == 'VisualCortex':
            self.kernel_scale = np.array([a[0] * ker_pre_norm, a[2] * ker_post_norm, a[3] * ker_post_post_norm])
        else:
            print('Wrong data_name!!!')
            return
            

