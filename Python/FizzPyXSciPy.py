#!/usr/bin/env python
# FizzPyX - FizzPyXSciPy
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from numpy import array, arange, size, empty, ndarray
from time import time
from math import exp, tanh, cosh
import scipy.integrate as integrate
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, savefig

from Python.FizzPyX import initIdentifier


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


# Main

if __name__ == '__main__':
    startTime1 = time()

    t = arange(0, 100, 0.02)
    zinit = initIdentifier('VDP')    #   [0.01, 0.01]
    z = integrate.odeint(vdp_scipy, zinit, t)
    xdot, ydot = z.T
    endTime1 = time()
    elapsedTime1 = (endTime1 - startTime1)

    figure()
    plot(t, xdot)
    # plot(t, y)
    title("My Soln")
    xlabel('Time')
    ylabel('Dynamical Variable')
    savefig('%s_tplot.png' % 'Cheese')

    print('The solver took ' + str(elapsedTime1) + ' seconds to execute. Which is faster than '
                                                  'I could do it on paper so we\'ll call it good.')
