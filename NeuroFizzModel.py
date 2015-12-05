#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzModel
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
import numpy as np
import math as mt

# Super class for all other models


class Model:
    def __init__(self, name, x0, t0, t1, dt):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'X Dynamical Variable'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self):
        pass

# van der Pol oscillator


class VDP(Model):
    def __init__(self, name='van der Pol oscillator', x0=np.array([0.01,0.01]), t0=0, t1=100, dt=0.02):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'X Dynamical Variable'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, mu=1):
        return np.array([x[1]/mu,
                         (-x[0] + x[1]*(1-x[0]**2))*mu])

# Leaky Integrate and Fire neuron model


class LIF(Model):
    def __init__(self, name='Leaky integrate-and-fire', x0=np.array([-65]), t0=0, t1=100, dt=0.1):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, rm=1, cm=10, tau_m=10, I = 1.5):
        return np.array([x[0] + (-x[0] + I*rm)/tau_m])

# Fitzhugh-Nagumo neuron model, supercritical Hopf bifurcation


class FN(Model):
    def __init__(self, name='Fitzhugh-Nagumo', x0=np.array([0.01, 0.01]), t0=0, t1=100, dt=0.02):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, a=0.75, b=0.8, c=3,  i=-0.40):
        return np.array([c*(x[0]+ x[1]- x[0]**3/3 + i),
                         -1/c*(x[0]- a + b*x[1])])

# Morris-Lecar neuron model, supercritical Hopf bifurcation


class ML(Model):
    def __init__(self, name='Morris-Lecar', x0=np.array([0, 0]), t0=0, t1=1000, dt=0.30):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, c=20, vk=-84, gk=8, vca=130, gca=4.4, vl=-60, gl=2, phi=0.04, v1=-1.2, v2=18, v3=2, v4=30, i=80):
        return np.array([(-gca*(0.5*(1 + mt.tanh((x[0] - v1)/v2)))*(x[0]-vca) - gk*x[1]*(x[0]-vk) - gl*(x[0]-vl) + i),
                        (phi*((0.5*(1 + mt.tanh((x[0] - v3)/v4))) - x[1]))/(1/mt.cosh((x[0] - v3)/(2*v4)))])

# Izhikevich neuron model, supercritical Hopf bifurcation


class IZ(Model):
    def __init__(self, name='Izhikevich', x0=np.array([0,0]), t0=0, t1=300, dt=0.1):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, a=0.02, b=0.2, c=-55, d=2, i=10):
        if x[0] >= 30:
            x[0] = c
            x[1] = x[1] + d
        return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + i,
                        a*(b*x[0] - x[1])])

# Hindmarsh-Rose neuron model, supercritical Hopf bifurcation


class HR(Model):
    def __init__(self, name='Hindmarsh-Rose', x0=np.array([3, 0, -1.2]), t0=0, t1=800, dt=0.1):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, I=1, xnot=-1.5):
        return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I,
                        c - d*(x[0]**2) - x[1],
                        r*(s*(x[0] - xnot) - x[2])])

# Hodgkin-Huxley neuron model


class HH(Model):
    def __init__(self, name='Hodgkin-Huxley', x0=np.array([0.01,0.01,0.01,0.01]), t0=0, t1=100, dt=0.02):
        self.name = name
        self.xaxis = 'Time'
        self.yaxis = 'Membrane Potential'
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_array = np.arange(t0, t1, dt)

    def eqns(self, x, t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
        alpha_n = (0.01*(x[0]+10))/(mt.exp((x[0]+10)/10)-1)
        beta_n = 0.125*mt.exp(x[0]/80)
        alpha_m = (0.1*(x[0]+25))/(mt.exp((x[0]+25)/10)-1)
        beta_m = 4*mt.exp(x[0]/18)
        alpha_h = (0.07*mt.exp(x[0]/20))
        beta_h = 1 / (mt.exp((x[0]+30)/10)+1)
        return np.array([(g_K*(x[1]**4)*(x[0]-E_K) + g_Na*(x[2]**3)*x[3]*(x[0]-E_Na) + g_L*(x[0]-E_L) - I)*(-1/C_m),
                        alpha_n*(1-x[1]) - beta_n*x[1],
                        alpha_m*(1-x[2]) - beta_m*x[2],
                        alpha_h*(1-x[3]) - beta_h*x[3]])

