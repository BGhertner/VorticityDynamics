# Ben Ghertner 2025
#
# Useful routines relating to Fourier methods
#
# 1) pad    - Pad a 2D array of fourier coefficients
#             used for interpolation and dealiasing
# 2) unpad   - Inversion of the pad function
#
# 3) interp - Fourier interpolation in 2D

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
    assert ( (Mx>=Nx)  and (My>=Ny)  )

    #allocate new array
    coefs_pad = np.zeros((My, Mx)).astype('complex128')

    #insert zeros
    coefs_pad[:Ny//2,  :Nx//2]  = coefs[:Ny//2,  :Nx//2]
    coefs_pad[-Ny//2:, :Nx//2]  = coefs[-Ny//2:, :Nx//2]
    coefs_pad[:Ny//2,  -Nx//2:] = coefs[:Ny//2,  -Nx//2:]
    coefs_pad[-Ny//2:, -Nx//2:] = coefs[-Ny//2:, -Nx//2:]

    return (Mx*My/Nx/Ny)*coefs_pad

"""
Inversion of the pad function (discard higher frequency components)

inputs:
    coefs_pad   - (2D numpy array) Original Fourier coefficients
    shape       - (tuple) New number of Fourier modes in x and y directions
                  Note: the new shape needs to be smaller than the original one.

returns:
    coefs       - (2D numpy array) Remaining Fourier coefficients
                  with the higher freqencies discarded.
"""
def unpad(coefs_pad, shape):
    
    #dimensions
    Ny, Nx = shape       #new
    My, Mx = coefs_pad.shape #original
    assert ( (Nx%2==0) and (Ny%2==0) )
    assert ( (Mx>=Nx)  and (My>=Ny)  )

    #allocate new array
    coefs = np.empty((Ny, Nx)).astype('complex128')

    #insert zeros
    coefs[:Ny//2,  :Nx//2]  = coefs_pad[:Ny//2,  :Nx//2]
    coefs[-Ny//2:, :Nx//2]  = coefs_pad[-Ny//2:, :Nx//2]
    coefs[:Ny//2,  -Nx//2:] = coefs_pad[:Ny//2,  -Nx//2:]
    coefs[-Ny//2:, -Nx//2:] = coefs_pad[-Ny//2:, -Nx//2:]

    #suppress nyquist mode
    coefs[Ny//2,:] = 0
    coefs[:,Nx//2] = 0

    return (Nx*Ny/Mx/My)*coefs

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