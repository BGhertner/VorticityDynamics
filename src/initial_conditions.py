

import numpy as np
from scipy.special import jn_zeros, jv


def lamb_dipole(Lx=2*np.pi, Ly=2*np.pi, Nx=32, Ny=32, U=1, R=1):


    k = jn_zeros(1, 1)[0]/R
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)

    xx, yy = np.meshgrid(x, y)

    r = np.sqrt((xx**2) + (yy**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        sin_theta = yy/r

    omega0 = np.where(r > R, 0, -k*2*U*sin_theta*jv(1, k*r)/(jv(0, k*R)))
    omega0[int(Ny/2),int(Nx/2)] = 0

    return omega0.astype('complex128')