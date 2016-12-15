#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzMath
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv2

from __future__ import division
import numpy as np
# import math as mt


# class Neuron:
#     def __init__(self):


def modelSelector(modelname):
    if modelname is None:
        raise TypeError
    if modelname == 'VDP':
        return 1
    elif modelname == 'LIF':
        return 2
    elif modelname == 'FN':
        return 3
    elif modelname == 'ML':
        return 4
    elif modelname == 'IZ':
        return 5
    elif modelname == 'HR':
        return 6
    elif modelname == 'HH':
        return 7
    else:
        return 0


def solverSelector(solvername):
    if solvername is None:
        raise TypeError
    if solvername == 'euler':
        return 1
    elif solvername == 'ord2':
        return 2
    elif solvername == 'rk4':
        return 3
    else:
        return 0


def Euler(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        X[i + 1] = X[i] + k1 * dt
    return X


def SecondOrder(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        k2 = model(X[i], tsp[i]) + k1 * (dt / 2)
        X[i + 1] = X[i] + k2 * dt
    return X


def RungeKutte4(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(1, nsize):
        k1 = model(X[i-1], tsp[i-1])
        k2 = model(X[i-1] + dt/2*k1, tsp[i-1] + dt/2)
        k3 = model(X[i-1] + dt/2*k2, tsp[i-1] + dt/2)
        k4 = model(X[i-1] + dt*k3, tsp[i-1] + dt)
        X[i] = X[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X


def VanDerPol(x,t):
    return np.array([x[1],
                    -x[0] + x[1]*(1-x[0]**2)])


def solutionGenerator(modelname, solvername):
    if int(modelSelector(modelname)) and modelSelector(modelname) in np.arange(1, 8):
        if int(solverSelector(solvername)) and solverSelector(solvername) in np.arange(1, 4):
            if solverSelector(solvername) == 1 and modelSelector(modelname) == 1:
                solution = Euler(x0=np.array([0.01, 0.01]), t1=100, dt=0.02, model=VanDerPol)
                return solution
            elif solverSelector(solvername) == 2 and modelSelector(modelname) == 1:
                solution = SecondOrder(x0=np.array([0.01, 0.01]), t1=100, dt=0.02, model=VanDerPol)
                return solution
            elif solverSelector(solvername) == 3 and modelSelector(modelname) == 1:
                solution = RungeKutte4(x0=np.array([0.01, 0.01]), t1=100, dt=0.01, model=VanDerPol)
                return solution
            else:
                solution = "Cheese please!"
                return solution


x = modelSelector('HH')

print(x)

y = solverSelector('rk4')

print(y)

z = solutionGenerator('VDP', 'rk4')

print(z)