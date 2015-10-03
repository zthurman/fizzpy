#!/usr/bin/env python
# NeuroFizzMath
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzmath import euler, ord2, rk4
from scipy import *
import numpy as np
import math as mt

# global solver methods:

# Euler solver (first order Runge-Kutta)

def euler(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):
    tsp = np.arange(t0, t1, dt)
    Nsize = np.size(tsp)
    X = np.empty((Nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, Nsize-1):
        k1 = ng(X[i],tsp[i])
        X[i+1] = X[i] + k1*dt
    return X

# second order solver (second order Runge-Kutta)

def ord2(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):
    tsp = np.arange(t0, t1, dt)
    Nsize = np.size(tsp)
    X = np.empty((Nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, Nsize-1):
        k1 = ng(X[i],tsp[i])
        k2 = ng(X[i],tsp[i]) + k1*(dt/2)
        X[i+1] = X[i] + k2*dt
    return X

# fourth order Runge-Kutte solver

def rk4(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):
    tsp = np.arange(t0, t1, dt)
    Nsize = np.size(tsp)
    X = np.empty((Nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, Nsize-1):
        k1 = ng(X[i],tsp[i])
        k2 = ng(X[i] + dt/2*k1, tsp[i] + dt/2)
        k3 = ng(X[i] + dt/2*k2, tsp[i] + dt/2)
        k4 = ng(X[i] + dt*k3, tsp[i] + dt)
        X[i+1] = X[i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X

# delay differential equation solver

def dde():
    pass