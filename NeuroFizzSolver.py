#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzSolver
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
import numpy as np
import time as tm

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
            k1 = self.model(self.X[i], self.tsp[i])
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
            k1 = self.model(self.X[i], self.tsp[i])
            k2 = self.model(self.X[i], self.tsp[i]) + k1*(self.dt/2)
            self.X[i+1] = self.X[i] + k2*self.dt
        return self.X

# fourth order Runge-Kutte solver

class rk4(Solver):

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
            k1 = self.model(self.X[i], self.tsp[i])
            k2 = self.model(self.X[i] + self.dt/2*k1, self.tsp[i] + self.dt/2)
            k3 = self.model(self.X[i] + self.dt/2*k2, self.tsp[i] + self.dt/2)
            k4 = self.model(self.X[i] + self.dt*k3, self.tsp[i] + self.dt)
            self.X[i+1] = self.X[i] + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return self.X

# Unit Test

def SolverTest():
    for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
        test = i
        print(test.name, test.x0, test.t0, test.t1, test.dt, test.t_array, test.eqns)
        for j in [euler(test.name, test.x0, test.dt, test.t_array, test.eqns),
                  ord2(test.name, test.x0, test.dt, test.t_array, test.eqns),
                  rk4(test.name, test.x0, test.dt, test.t_array, test.eqns)]:
            soln = j
            print(soln.tsp, soln.Nsize, soln.X, soln.X[0], soln.X[:,0], soln.evaluate())

# Time execution of the solvertest

starttime = tm.time()
solvertester = SolverTest()
print(" %s seconds" % (tm.time() - starttime))