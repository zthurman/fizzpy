#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzSolver
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, IZ, HR, HH, RD, L, R
from scipy import *
import numpy as np
import math as mt

# global solver class

class Solver():

    # Solver class variables

    def __init__(self, name, x0, dt, t_array, eqns):
        self.modelname = name
        self.tsp = t_array
        self.dt = dt
        self.Nsize = np.size(self.tsp)
        self.X = np.empty((self.Nsize, np.size(x0)))
        self.X[0] = x0
        self.model = eqns

    def evaluate(self):
        pass

    # Euler solver (first order Runge-Kutta)

"""    def euler(Solver, t0 = Model.t0, t1 = Model.t1, dt = Model.dt, x0 = Model.x0, Model = Model.eqns):
        tsp = np.arange(t0, t1, dt)
        Nsize = np.size(tsp)
        X = np.empty((Nsize, np.size(x0)))
        X[0] = x0
        for i in range(0, Solver.Nsize-1):
            k1 = Model(Solver.X[i],tsp[i])
            Solver.X[i+1] = Solver.X[i] + k1*dt
        return X

    # second order solver (second order Runge-Kutta)

    def ord2(Solver, t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, Model = None):
        tsp = np.arange(t0, t1, dt)
        Nsize = np.size(tsp)
        X = np.empty((Nsize, np.size(x0)))
        X[0] = x0
        for i in range(0, Nsize-1):
            k1 = Model(X[i],tsp[i])
            k2 = Model(X[i],tsp[i]) + k1*(dt/2)
            X[i+1] = X[i] + k2*dt
        return X

    # fourth order Runge-Kutte solver

    def rk4(Solver, t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, Model = None):
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

    def dde(Solver):
        pass"""

# Euler solver (first order Runge-Kutta)

class euler(Solver):
    def __init__(self, name, x0, dt, t_array, eqns):
        self.modelname = name
        self.dt = dt
        self.tsp = t_array
        self.Nsize = np.size(self.tsp)
        self.X = np.empty((self.Nsize, np.size(x0)))
        self.X[0] = x0
        self.model = eqns

    def evaluate(self):
        for i in range(0, self.Nsize-1):
            k1 = self.eqns(self.X[i], self.tsp[i])
            self.X[i+1] = self.X[i] + k1*self.dt
        return self.X

# second order solver (second order Runge-Kutta)

class ord2(Solver):

    def __init__(self, name, x0, dt, t_array, eqns):
        self.modelname = name
        self.dt = dt
        self.tsp = t_array
        self.Nsize = np.size(self.tsp)
        self.X = np.empty((self.Nsize, np.size(x0)))
        self.X[0] = x0
        self.model = eqns

    def evaluate(self):
        for i in range(0, self.Nsize-1):
            k1 = self.eqns(self.X[i], self.tsp[i])
            k2 = self.eqns(self.X[i], self.tsp[i]) + k1*(self.dt/2)
            self.X[i+1] = self.X[i] + k2*self.dt
        return self.X

# debug
test = VDP()

soln = euler(test.name, test.x0, test.dt, test.t_array, test.eqns)
# debug
print(soln.tsp)
# debug
print(soln.Nsize)
# debug
print(soln.X)
# debug
print(soln.X[0])
# debug
print(soln.X[:,0])
