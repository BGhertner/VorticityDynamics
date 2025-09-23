
import numpy as np
import scipy.fft as sfft

from poisson import poisson


def vort_rhs(omega, Lx, Ly, Nx, Ny, Uinf):

    #Wave numbers
    k = sfft.fftfreq(Nx)*Nx*2*np.pi/Lx 
    l = sfft.fftfreq(Ny)*Ny*2*np.pi/Ly
    kk, ll = np.meshgrid(k, l)

    #Spatial grid
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    _, yy = np.meshgrid(x, y)

    #Reshape omega to 2D array
    omega_2D = np.reshape(omega, (Nx, Ny))
    A = poisson(omega_2D, Lx, Ly, Nx, Ny, Uinf, Fourier=True)

    #Compute velocity components
    u = -1j*sfft.ifft2(A*ll) + Uinf
    v =  1j*sfft.ifft2(A*kk)

    #Compute (omega x u)_x
    B = sfft.fft2(omega_2D*u)
    assert ( (Nx%2==0) and (Ny%2==0) )
    B[int(Nx/2),:] = 0
    B[:,int(Ny/2)] = 0
    omegau_x = 1j*sfft.ifft2(B*kk)

    #Compute (omega x v)_y
    C = sfft.fft2(omega_2D*v)
    assert ( (Nx%2==0) and (Ny%2==0) )
    C[int(Nx/2),:] = 0
    C[:,int(Ny/2)] = 0
    omegav_y = 1j*sfft.ifft2(C*ll)

    #return "flattened" 1D right hand side
    return (-omegau_x - omegav_y).flatten()
