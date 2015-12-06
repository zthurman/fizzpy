#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzSolver
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
import numpy as np

# Super class for all other solvers


class Solver:
    def __init__(self, name, x0, dt, t_array, equations):
        self.model_name = name
        self.tsp = t_array
        self.dt = dt
        self.N_size = np.size(self.tsp)
        self.X = np.empty((self.N_size, np.size(x0)))
        self.X[0] = x0
        self.model = equations

    def evaluate(self):
        pass

# Euler solver (first order Runge-Kutta)


class euler(Solver):
    def __init__(self, model_name, xaxis, yaxis, x0, dt, t_array, equations):
        self.name = 'Euler'
        self.model_name = model_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.dt = dt
        self.tsp = t_array
        self.N_size = np.size(self.tsp)
        self.X = np.empty((self.N_size, np.size(x0)))
        self.X[0] = x0
        self.model = equations

    def evaluate(self):
        for i in range(0, self.N_size-1):
            k1 = self.model(self.X[i], self.tsp[i])
            self.X[i+1] = self.X[i] + k1*self.dt
            return self.X

# second order solver (second order Runge-Kutta)


class ord2(Solver):
    def __init__(self, model_name, xaxis, yaxis, x0, dt, t_array, equations):
        self.name = 'Second Order Runge-Kutta'
        self.model_name = model_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.dt = dt
        self.tsp = t_array
        self.N_size = np.size(self.tsp)
        self.X = np.empty((self.N_size, np.size(x0)))
        self.X[0] = x0
        self.model = equations

    def evaluate(self):
        for i in range(0, self.N_size-1):
            k1 = self.model(self.X[i], self.tsp[i])
            k2 = self.model(self.X[i], self.tsp[i]) + k1*(self.dt/2)
            self.X[i+1] = self.X[i] + k2*self.dt
        return self.X

# fourth order Runge-Kutta solver


class rk4(Solver):
    def __init__(self, model_name, xaxis, yaxis, x0, dt, t_array, equations):
        self.name = 'Fourth Order Runge-Kutta'
        self.model_name = model_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.dt = dt
        self.tsp = t_array
        self.N_size = np.size(self.tsp)
        self.X = np.empty((self.N_size, np.size(x0)))
        self.X[0] = x0
        self.model = equations

    def evaluate(self):
        for i in range(0, self.N_size-1):
            k1 = self.model(self.X[i], self.tsp[i])
            k2 = self.model(self.X[i] + self.dt/2*k1, self.tsp[i] + self.dt/2)
            k3 = self.model(self.X[i] + self.dt/2*k2, self.tsp[i] + self.dt/2)
            k4 = self.model(self.X[i] + self.dt*k3, self.tsp[i] + self.dt)
            self.X[i+1] = self.X[i] + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return self.X

