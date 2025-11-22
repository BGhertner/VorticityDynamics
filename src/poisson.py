# Ben Ghertner 2025
#
# Solve the Poisson equation relating the vorticity and streamfunction.
# Uses Fourier spectral methods.

import numpy as np
import scipy.fft as sfft

"""
Solve the Poisson equation relating the vorticity and streamfunction.
    omega   - (2D numpy array with dimensions Ny x Nx) Vorticity
    Lx      - (float) Horizontal length of the domain
    Ly      - (float) Vertical length of the domain
    Nx      - (int) Number of grid points in the horizontal direction
    Ny      - (int) Number of grid points in the vertical direction
    Uinf    - (float) Horizontal background wind speed
    Fourier - (bool) If true return the fourier coefficients for the stream function
                     If false return the stream function on the spatial grid

returns:
    psi     - (2D numpy array with dimensions Ny x Nx) Streamfunction
              either in Fourier space or on the spatial grid
"""
def poisson(omega, Lx, Ly, Nx, Ny, Uinf=0., Fourier=False):
    
    #Wave numbers
    k = sfft.fftfreq(Nx)*Nx*2*np.pi/Lx 
    l = sfft.fftfreq(Ny)*Ny*2*np.pi/Ly
    kk, ll = np.meshgrid(k, l)

    #reciprocal of total wave number squared
    with np.errstate(divide='ignore', invalid='ignore'):
        mu2_r = 1/(kk**2 + ll**2)
    mu2_r[0,0] = 0.

    #Fourier coefficients for psi
    A = -sfft.fft2(omega)*mu2_r

    #0 out nyquist modes
    assert ( (Nx%2==0) and (Ny%2==0) )
    A[int(Ny/2),:] = 0
    A[:,int(Nx/2)] = 0

    if Fourier: return A

    #Spatial grid
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    _, yy = np.meshgrid(x, y)

    psi = sfft.ifft2(A) - Uinf*yy

    return psi