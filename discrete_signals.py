"""Some functions for constructing/manipulating discrete signals."""

import jax.numpy as np
from jax import vmap
from jax.lax import conv_general_dilated
from jax.ops import index_update, index
from numpy.polynomial.legendre import Legendre

def discretized_legendre(degree, N, ran=[-1,1]):
    """
    Discretized Legendre polynomial of specified degree over interval [0,T], with samples taken
    using midpoint rule

    Args:
        degree (int): degree of Legendre polynomial (starting from 0)
        T (float): right limit of domain [0,T]
        N (int): number of steps in discretization
        ran (list): upper and lower bound on range of polynomial

    """
    dt = 1. / N
    coeffs = np.zeros(degree + 1)
    coeffs = index_update(coeffs, index[-1], 1.)

    return np.array(Legendre(coeffs, [0,1.], ran)(np.linspace(0, 1. - dt, N) + (dt / 2)))

def discretized_legendre_basis(max_degree, N, ran=[-1,1]):
    """Return all discretized legendre polynomials up to a given degree
    """

    # Note: I'd like to do the following line, but there is some jax issue
    #return vmap(lambda deg: discretized_legendre(deg, T, N, ran))(np.array(list(range(max_degree+1))))

    vals = list(range(max_degree + 1))
    disc_leg_map = map(lambda deg: discretized_legendre(deg, N, ran), vals)

    # convert to array and return
    return np.array(list(disc_leg_map))

def get_diffeo(image=[-1, 1]):
    """
    return a diffeomorphism from R into image
    """

    width = image[1] - image[0]

    return lambda x: np.arctan(x)/(np.pi/2) * (width / 2) + image[0] + (width / 2)


def fine_1d_conv(in_x, kernel, kern_sample_rate=1):
    """Convolve input with a kernel, with kern_sample_rate giving the number of samples in
    kernel per sample of in_x.

    This function "expands" in_x by repeating each entry in in_x kern_sample_rate times,
    then convolves the expanded signal with kernel, padding with len(kernel) zeros. The purpose
    of the padding is to ensure that the convolved signal starts and ends with 0.

    Args:
        in_x (1d array): input signal
        kernel (1d array): kernel
        kern_sample_rate (int): number

    Returns:
        array: convolved signal
    """

    # expand the input
    # this may actually be doable with conv_general_dilated
    expanded = vmap(lambda x: x * np.ones(kern_sample_rate))(in_x).flatten()
    kernel_len = len(kernel)

    return conv_general_dilated(np.array([[expanded]]),
                                np.array([[kernel]]),
                                (1,),
                                padding=[(kernel_len,kernel_len)] )[0,0]

# gaussian function
def gaussian(x, amp, mean, sig):
    """compute value of a gaussian with certain parameters"""
    return amp * np.exp(-np.power(x - mean, 2.) / (2 * np.power(sig, 2.)))

def gauss_conv_kernel(amp, sigma, samples, width):
    """constructs a discrete gaussian"""
    return vmap(lambda t: gaussian(t, amp, 0, sigma))(np.linspace(-width, width, samples))

def get_gaussian_filter(samples, zero_tol=10**-4, kern_sample_rate=1):
    """Returns a function that convolves a 1d array with a discrete gaussian.

    The function automatically chooses the parameters of the gaussian so that the endpoint samples
    are below zero_tol, and the sum of the samples is 1.

    Args:
        samples (int): number of samples in the discrete gaussian
        zero_tol (float): set upper bound for endpoint samples (this is before normalization)
        kern_sample_rate (int): For the returned function, number of samples in kernel per
                                sample in the input array

    Returns:
        function implementing specified convolution
    """

    # get x value for which gaussian of amp=1, mean=0, and std=sigma hits zero_tol
    width = np.sqrt(- (2 * np.power(1., 2.)) * np.log(zero_tol))

    conv_kernel = gauss_conv_kernel(1., 1., samples, width)
    conv_kernel = conv_kernel / np.sum(conv_kernel)

    return lambda in_x: fine_1d_conv(in_x, conv_kernel, kern_sample_rate)

def get_param_to_signal(leg_order, N):
    
    dt_awg = 1. # length of time step for piecewise constant signal
    filter_std = 0.5 # std of convolved gaussian

    # derived parameters
    filter_samples_per_awg = int(4/filter_std) # number of samples of gaussian per sample of piecewise constant signal
    dt_filter = dt_awg / filter_samples_per_awg
    N_filter = int(np.ceil( (3 * filter_std) / dt_filter )) # number of samples for the filter
    filter_time_width = N_filter * dt_filter
    gaussian = gauss_conv_kernel(1., filter_std, 2*N_filter, filter_time_width)
    gauss_conv = gaussian / gaussian.sum()

    # Function that applies filter to an array representing a piecewise constant filter
    apply_filter = lambda in_sig: fine_1d_conv(in_sig, gauss_conv, kern_sample_rate=filter_samples_per_awg)


    
    # create basis
    signal_basis = discretized_legendre_basis(leg_order, N)
    
    # create diffeomorphism from R into [-1, 1]
    diffeo = get_diffeo([-1, 1])
    
    def param_to_signal(params):
        # take linear combo of discretized legendre polynomials
        leg_linear_combo = np.tensordot(params, signal_basis, axes=1)
    
        # map all samples into [-1,1]
        normalized = vmap(lambda x: diffeo(x), in_axes=1, out_axes=1)(leg_linear_combo)
    
        # convolve both X and Y signals, then transpose 
        return vmap(apply_filter)(normalized.real).transpose() + 0*1j
    
    return param_to_signal