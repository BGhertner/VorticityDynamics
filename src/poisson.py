import numpy as np
import scipy.fft as sfft

def poisson(omega, Lx, Ly, Nx, Ny, Uinf=0., Fourier=False):
    
    #Wave numbers
    k = sfft.fftfreq(Nx)*Nx*2*np.pi/Lx 
    l = sfft.fftfreq(Ny)*Ny*2*np.pi/Ly
    kk, ll = np.meshgrid(k, l)

    #reciprocal of total wave number squared
    mu2_r = 1/(kk**2 + ll**2)
    mu2_r[0,0] = 0.

    #Fourier coefficients for psi
    A = -sfft.fft2(omega)*mu2_r

    #0 out nyquist modes
    assert ( (Nx%2==0) and (Ny%2==0) )
    A[int(Nx/2),:] = 0
    A[:,int(Ny/2)] = 0

    if Fourier: return A

    #Spatial grid
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    _, yy = np.meshgrid(x, y)

    psi = sfft.ifft2(A) + Uinf*yy

    return psi