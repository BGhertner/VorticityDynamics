# Ben Ghertner 2025
#
# Preform time-stepping and generate a matplotlib animation of the solution.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.integrate import solve_ivp

from poisson import poisson
from vort_rhs import vort_rhs

"""
Preform time-stepping and generate a matplotlib animation of the solution.
Time-stepping is done with the RK23 adaptive explicit method.

inputs:
    om0     - (2D numpy array with dimensions Ny x Nx) Initial conditions 
              for the vorticity
    dt      - (float) time step between frames in the animation
    Lx      - (float) Horizontal length of the domain
    Ly      - (float) Vertical length of the domain
    Nx      - (int) Number of grid points in the horizontal direction
    Ny      - (int) Number of grid points in the vertical direction
    Uinf    - (float) Horizontal background wind speed
    fps     - (int) Frames per second in the animation
    frames  - (int) Number of frames in the animation 
              (this also dictates the length of the simulation)
    delay   - (float) Seconds paused at the end of the animation before
              it repeates itself

returns:
    Ani     - (matplotlib animation object)
"""
def animation(om0, dt=0.05, Lx=2*np.pi, Ly=2*np.pi, Nx=32, Ny=32, Uinf=1,
              fps=30, frames=100, delay=0.5):

    #Include periodic points for plotting
    om0_perpnt = np.empty((om0.shape[0]+1,om0.shape[1]+1))
    om0_perpnt[:-1, :-1] = om0.real
    om0_perpnt[-1,:-1] = om0[0,:].real
    om0_perpnt[:-1,-1] = om0[:,0].real
    om0_perpnt[-1,-1]  = om0[0,0].real

    #Grid with periodic points
    x = np.linspace(-Lx/2, Lx/2, Nx+1, endpoint=True)
    y = np.linspace(-Ly/2, Ly/2, Ny+1, endpoint=True)
    xx, yy = np.meshgrid(x, y)

    ommax = np.max(np.abs(om0))*(1.1)

    psi = poisson(om0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Uinf=Uinf).real
    psi_perpnt = np.empty((psi.shape[0]+1,psi.shape[1]+1))
    psi_perpnt[:-1, :-1] = psi
    psi_perpnt[-1,:-1] = psi[0,:] - y[0]*Uinf + y[-1]*Uinf
    psi_perpnt[:-1,-1] = psi[:,0]
    psi_perpnt[-1,-1] =  psi[0,0] - y[0]*Uinf + y[-1]*Uinf

    psimax = np.max(np.abs(psi))*(1.1)

    #Set up plot for animation
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5)
    fig.set_dpi(200)

    contf = ax.contourf(xx, yy, om0_perpnt, levels=np.linspace(-ommax, ommax, num=12), cmap='PRGn')
    contp = ax.contour(xx, yy, psi_perpnt, levels=np.linspace(0,psimax, num=7)[1:], colors='k', linewidths=1)
    cont0 = ax.contour(xx, yy, psi_perpnt, levels=[0], colors='b', linewidths=1)
    contn = ax.contour(xx, yy, psi_perpnt, levels=np.linspace(-psimax, 0, num=7)[:-1], colors='k', linewidths=1)
    ax.set_aspect(1.0)
    ax.set(xlim=(-Lx/2,Lx/2), 
        ylim=(-Ly/2,Ly/2), 
        xlabel=r'$x$', 
        ylabel=r'$y$', 
        title=r'Vorticity Dynamics in a Periodic Box')
    fig.colorbar(contf, label=r'$\omega$', shrink=0.8, ticks=(-10, -5, 0, 5, 10))

    plots = {'contf':contf, 
            'contp':contp,
            'cont0':cont0,
            'contn':contn}

    #initial condition
    global om
    om = om0

    def animate(i):
        global om

        if i == 0: om = om0
        
        if i <= frames:
            #clear old plot elements
            plots['contf'].remove()
            plots['contp'].remove()
            plots['cont0'].remove()
            plots['contn'].remove()


            #Preform time step
            sol = solve_ivp(lambda t, y: vort_rhs(y, Lx, Ly, Nx, Ny, Uinf),
                        (0.0, dt),
                        om.flatten(),
                        method='RK23',
                        atol=1e-10, rtol=1e-10)

            om = np.reshape(sol.y[:,-1], (Ny,Nx))

            #Include periodic points for plotting
            om_perpnt = np.empty((om.shape[0]+1,om.shape[1]+1))
            om_perpnt[:-1, :-1] = om.real
            om_perpnt[-1,:-1] = om[0,:].real
            om_perpnt[:-1,-1] = om[:,0].real
            om_perpnt[-1,-1]  = om[0,0].real

            psi = poisson(om, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Uinf=Uinf).real
            psi_perpnt = np.empty((psi.shape[0]+1,psi.shape[1]+1))
            psi_perpnt[:-1, :-1] = psi
            psi_perpnt[-1,:-1] = psi[0,:] - y[0]*Uinf + y[-1]*Uinf
            psi_perpnt[:-1,-1] = psi[:,0]
            psi_perpnt[-1,-1] =  psi[0,0] - y[0]*Uinf + y[-1]*Uinf

            #Redraw plot
            plots['contf'] = ax.contourf(xx, yy, om_perpnt, levels=np.linspace(-ommax, ommax, num=12), cmap='PRGn')
            plots['contp'] = ax.contour(xx, yy, psi_perpnt, levels=np.linspace(0,psimax, num=7)[1:], colors='k', linewidths=1)
            plots['cont0'] = ax.contour(xx, yy, psi_perpnt, levels=[0], colors='b', linewidths=1)
            plots['contn'] = ax.contour(xx, yy, psi_perpnt, levels=np.linspace(-psimax, 0, num=7)[:-1], colors='k', linewidths=1)

        return plots

    return ani.FuncAnimation(fig=fig, func=animate, blit=False, interval=1000/fps, 
                             frames=frames+int(delay*fps), repeat=False)