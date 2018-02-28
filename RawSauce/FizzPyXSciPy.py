#!/usr/bin/env python
# FizzPyX - FizzPyXSciPy
# Copyright (C) 2017 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from numpy import array, arange, size, empty, ndarray
from time import time
from math import exp, tanh, cosh
import scipy.integrate as integrate
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, savefig

from Numpy_Neurons.FizzPyX import initIdentifier


# Scipy Implementations

# Van Der Pol

def vdp_scipy(z, t):
    xdot, ydot = z
    mu = 1
    return [ydot, mu*ydot*(1-xdot**2)-xdot]


# Damped SHM

def shm_scipy(z, t):
    xdot, ydot = z
    r = 0.035
    s = 0.5
    m = 0.2
    return array([ydot, (-r*ydot - s*xdot)/m])


# Coupled Oscillating Carts

def co_scipy(z, t):
    x, xdot, y, ydot = z
    b = 0.007
    k1 = 0.27
    k2 = 0.027
    m = 0.25
    return array([xdot, -(k1/m)*x + (k2/m)*y - (b/m)*xdot, ydot, (k2/m)*x - (k1/m)*y - (b/m)*ydot])


# Lorenz attractor

def lo_scipy(z, t):
    xdot, ydot, zdot = z
    sigma = 10.0
    rho = 28.0
    beta = 10.0/3
    return array([sigma * (ydot - xdot), rho*xdot - ydot - xdot*zdot, xdot*ydot - beta*zdot])


# Robbins model

def rb_scipy(z, t):
    xdot, ydot, zdot = z
    V = 1
    sigma = 5
    R = 13
    return array([R - ydot*zdot - V*xdot, xdot*zdot - ydot, sigma*(ydot - zdot)])


# Rikitake Dynamo

def ri_scipy(z, t):
    i, idot, j, jdot, k, kdot = z
    m = 0.5
    g = 50
    r = 8
    f = 0.5
    return array([r*(jdot - i),
                 r*(j - idot),
                 i*k + m*idot - (1 + m)*j,
                 idot*kdot + m*i - (1 + m)*jdot,
                 g*(1 - (1 + m)*i*j + m*i*idot) - f*k,
                 g*(1 - (1 + m)*idot*jdot + m*idot*i) - f*kdot])


# Main

if __name__ == '__main__':
    startTime1 = time()

    t = arange(0, 100, 0.02)
    zinit = initIdentifier('RI')    #   [0.01, 0.01]
    z = integrate.odeint(ri_scipy, zinit, t)
    i, idot, j, jdot, k, kdot = z.T
    endTime1 = time()
    elapsedTime1 = (endTime1 - startTime1)

    figure()
    plot(t, idot)
    # plot(t, y)
    title("My Soln")
    xlabel('Time')
    ylabel('Dynamical Variable')
    savefig('%s_tplot.png' % 'Cheese')

    print('The solver took ' + str(elapsedTime1) + ' seconds to execute. Which is faster than '
                                                  'I could do it on paper so we\'ll call it good.')
