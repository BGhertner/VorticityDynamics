# Ben Ghertner 2025
#
# Useful routines relating to Fourier methods
#
# 1) pad    - Pad a 2D array of fourier coefficients
#             used for interpolation and dealiasing
#
# 2) interp - Fourier interpolation in 2D

import numpy as np
import scipy.fft as sfft

"""
Pad a 2D array of fourier coefficients

inputs:
    coefs       - (2D numpy array) Original Fourier coefficients
    shape       - (tuple) New number of Fourier modes in x and y directions
                  Note: the new shape needs to be larger than the original one.

returns:
    coefs_pad   - (2D numpy array) Padded Fourier coefficients
"""
def pad(coefs, shape):
    
    #dimensions
    Ny, Nx = coefs.shape #original
    My, Mx = shape       #new
    assert ( (Nx%2==0) and (Ny%2==0) )

    #allocate new array
    coefs_pad = np.zeros((My, Mx)).astype('complex128')

    #insert zeros
    coefs_pad[:Ny//2,  :Nx//2]  = coefs[:Ny//2,  :Nx//2]
    coefs_pad[-Ny//2:, :Nx//2]  = coefs[-Ny//2:, :Nx//2]
    coefs_pad[:Ny//2,  -Nx//2:] = coefs[:Ny//2,  -Nx//2:]
    coefs_pad[-Ny//2:, -Nx//2:] = coefs[-Ny//2:, -Nx//2:]

    return coefs_pad

"""
Fourier interpolation in 2D. Interpolate a function to a denser grid.

inputs:
    f           - (2D numpy array) Function on the original grid.
    shape       - (tuple) New number of grid points in the x and y directions

returns:
    f_interp    - (2D numpy array) Function on the new dense grid.
"""
def interp(f, shape):
    coefs = sfft.fft2(f)
    coefs_pad = pad(coefs, shape)
    return sfft.ifft2(coefs_pad)