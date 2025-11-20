# Ben Ghertner 2025
#
# Right hand side function for time stepping of the vorticity equation.
#
# 1) Solve the Poisson equation relating the vorticity and stream function.
# 2) Use spectral differentiation to compute the velocity components u and v.
# 3) Use spectral differentiation to compute derivatives of the nonlinear terms.

import numpy as np
import scipy.fft as sfft

from poisson import poisson
from fourier import interp, unpad

"""
Right hand side function for time stepping of the vorticity equation.

inputs:
    omega   - (1D numpy array with length Nx x Ny) Vorticity
    Lx      - (float) Horizontal length of the domain
    Ly      - (float) Vertical length of the domain
    Nx      - (int) Number of grid points in the horizontal direction
    Ny      - (int) Number of grid points in the vertical direction
    Uinf    - (float) Horizontal background wind speed

returns:
    omega_t - (1D numpy array with length Nx x Ny) 
              Time derivative of the vorticity to be used in time stepping
"""
def vort_rhs(omega, Lx, Ly, Nx, Ny, Uinf):

    #Wave numbers
    k = sfft.fftfreq(Nx)*Nx*2*np.pi/Lx 
    l = sfft.fftfreq(Ny)*Ny*2*np.pi/Ly
    kk, ll = np.meshgrid(k, l)

    #Reshape omega to 2D array
    omega_2D = np.reshape(omega, (Ny, Nx))
    A = poisson(omega_2D, Lx, Ly, Nx, Ny, Uinf, Fourier=True)

    #Compute velocity components
    u = -1j*sfft.ifft2(A*ll) + Uinf
    v =  1j*sfft.ifft2(A*kk)

    ### Compute (omega x u)_x
    #de-aliasing
    omega_2D_pad = interp(omega_2D, (3*Ny//2, 3*Nx//2))
    u_pad        = interp(u,        (3*Ny//2, 3*Nx//2))
    omegau_pad = omega_2D_pad*u_pad
    B = unpad(sfft.fft2(omegau_pad), (Ny,Nx))
    #spectral derivative
    omegau_x = 1j*sfft.ifft2(B*kk)

    ### Compute (omega x v)_y
    #de-aliasing
    v_pad = interp(v, (3*Ny//2, 3*Nx//2))
    omegav_pad = omega_2D_pad*v_pad
    C = unpad(sfft.fft2(omegav_pad), (Ny,Nx))
    #spectral derivative
    omegav_y = 1j*sfft.ifft2(C*ll)

    #return "flattened" 1D right hand side
    return (-omegau_x - omegav_y).flatten()
