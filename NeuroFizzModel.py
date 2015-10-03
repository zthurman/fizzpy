#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzModel
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from scipy import *
import numpy as np
import math as mt

# System, super class for all models

class Model(object):

    # System class variables

    def __init__(self, name, x0, t_array):
        self.name = name
        self.x0 = x0
        self.t_array = t_array

    # van der Pol oscillator

    def VDP(self, name = 'van der Pol oscillator', x0 = np.array([0.01,0.01]), t_array = np.arange(0, 100, 0.02)):
        def model(self,x,t, mu = 1):
            return np.array([x[1]/mu,
                             (-x[0] + x[1]*(1-x[0]**2))*mu])

    # EPSP - excitatory post-synaptic potential

"""   def EPSP(System):
        name = "EPSP"
        x0 = np.array([0,0,0])

        def model(self,x,t):
            # inits and contants
            c_m = 1
            g_L = 1
            tau_syn = 1
            E_syn = 10
            delta_t = 0.01
            g_syn = np.zeros([1200])
            I_syn = np.zeros([1200])
            v_m = np.zeros([1200])
            t = np.zeros([1200])
            g_syn[0] = 0
            I_syn[0] = 0
            v_m[0] = 0
            t[0] = 0

            # Numerical integration with Euler's method
            for i in np.arange(0, (10/delta_t)):
                t[int(i+1)] = t[int((i))]+delta_t
                if np.abs((t[int(i+1)-1]))<0.001:
                    g_syn[int((i))] = 1

                g_syn[int(i+1)] = g_syn[int((i))]-np.dot((delta_t/tau_syn), g_syn[int((i))])
                I_syn[int(i+1)] = np.dot(g_syn[int(i+1)], v_m[int((i))]-E_syn)
                v_m[int(i+1)] = v_m[int((i))]-np.dot(np.dot((delta_t/c_m),g_L), v_m[int((i))]) \
                                -np.dot((delta_t/c_m), I_syn[int(i+1)])

    # Fitzhugh-Nagumo neuron model

    def FN(System, name = 'Fitzhugh-Nagumo', x0 = np.array([0.01,0.01]), t_array = np.arange(0, 100, 0.02)):
        def model(self,x,t, a = 0.75, b = 0.8, c = 3,  i = -0.40):
            return np.array([c*(x[0]+ x[1]- x[0]**3/3 + i),
                             -1/c*(x[0]- a + b*x[1])])

    # Morris-Lecar neuron model

    def ML(System, name = 'Morris-Lecar', x0 = np.array([0,0]), t_array = (0, 1000, 0.30)):
        def model(self,x,t,c = 20,vk=-84,gk = 8,vca = 130,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,i = 80):
            return np.array([(-gca*(0.5*(1 + mt.tanh((x[0] - v1)/v2)))*(x[0]-vca) - gk*x[1]*(x[0]-vk) - gl*(x[0]-vl) + i),
                            (phi*((0.5*(1 + mt.tanh((x[0] - v3)/v4))) - x[1]))/(1/mt.cosh((x[0] - v3)/(2*v4)))])

    # Izhikevich neuron model

    def IZ(System, name = 'Izhikevich', x0 = np.array([0,0]), t_array = np.arange(0, 300, 0.1)):
        def model(self,x,t, a = 0.02, b = 0.2, c = -55, d = 2, i = 10):
            if x[0] >= 30:
                x[0] = c
                x[1] = x[1] + d
            return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + i,
                            a*(b*x[0] - x[1])])

    # Hindmarsh-Rose neuron model

    def HR(System, name = 'Hindmarsh-Rose', x0 = np.array([3, 0, -1.2]), t_array = np.arange(0, 800, 0.1)):
        def model(self,x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 1, xnot = -1.5):
            return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I,
                            c - d*(x[0]**2) - x[1],
                            r*(s*(x[0] - xnot) - x[2])])

    # Hodgkins-Huxley neuron model

    def HH(System, name = 'Hodgkins-Huxley', x0 = np.array([0.01,0.01,0.01,0.01]), t_array = np.arange(0, 100, 0.02)):
        def model(self,x,t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
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

    # Rikitake dynamo model for geomagnetic reversal

    def RD(System, name = 'Rikitake Dynamo', x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t_array = np.arange(0, 100, 0.01)):
        def model(self, x,t, m = 0.5, g = 50, r = 8, f = 0.5):
            return np.array([r*(x[3] - x[0]),
                             r*(x[2] - x[1]),
                             x[0]*x[4] + m*x[1] - (1 + m)*x[2],
                             x[1]*x[5] + m*x[0] - (1 + m)*x[3],
                             g*(1 - (1 + m)*x[0]*x[2] + m*x[0]*x[1]) - f*x[4],
                             g*(1 - (1 + m)*x[1]*x[3] + m*x[1]*x[0]) - f*x[5]])

    # Hugh Wilson neuron model

    def W(System):
        name = 'Wilson Model'
        x0 = []

        def model(self, x, t):
            pass

        def Wilson(x,t, g_K=26, g_T=0.1, g_H=5, E_K=-0.95, E_Na=0.50, E_Ca=1.20, C=0.01, E_H=-0.95, tau_T=14, tau_R=4.2, tau_H=45, I=1):
                G_K = g_K*x[1]*100
                G_Na = 17.8 + 47.6*x[0] + 33.8*(x[0]**2)
                G_Ca = g_T*x[2]*100
                G_H = g_H*x[3]*100
                Rnot = 1.24 + (3.7*x[0]) + 3.20*(x[0]**2)
                Tnot = 4.205 + (11.6*x[0]) + 8*(x[0]**2)
                return np.array([(-G_Na*(x[0]-(E_Na/100)) - G_K*(x[0]-(E_K/100)) - G_Ca*(x[0]-(E_Ca/100)) - G_H*(x[0]-(E_H/100)) + I)*(C),
                                  -(x[1] - Rnot)*(1/tau_R),
                                  -(x[2] - Tnot)*(1/tau_T),
                                  -(x[3] - 3*x[2])*(1/tau_H)])

    # Lorenz atmospheric model, strange attractor

    def L(System, name = 'Lorenz Equations', x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t_array = np.arange(0, 100, 0.01)):
        def model(self, x, t, sigma = 10.0, rho = 28.0, beta = 10.0/3):
            return np.array([sigma * (x[1] - x[0]),
                             rho*x[0] - x[1] - x[0]*x[2],
                             x[0]*x[1] - beta*x[2]])

    # Robbins geomagnetic polarity reversal model

    def R(System, name = 'Robbins Equations', x0 = np.array([0.00032,0.23,0.51]), t_array = np.arange(0, 200, 0.1)):
        def model(self,x,t, V = 1, sigma = 5, R = 13):
            return np.array([R - x[1]*x[2] - V*x[0],
                             x[0]*x[2] - x[1],
                             sigma*(x[1] - x[2])])"""

# van der Pol oscillator

class VDP(object):
    def __init__(self):
        self.model = Model(name = 'van der Pol oscillator', x0 = np.array([0.01,0.01]), t_array = np.arange(0, 100, 0.02))

    def eqns(self,x,t, mu = 1):
        return np.array([x[1]/mu,
                         (-x[0] + x[1]*(1-x[0]**2))*mu])

test = VDP()
test.name
