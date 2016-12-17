#!/usr/bin/env python
# NeuroFizzMath - AlternateModels
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

# Here are all of the different models that are either not fully implemented as a numerically integrated
# system of equations (EPSP, Wilson) or are fully implemented but fall outside the scope of neurons and
# AI systems (Rikitake Dynamo, Lorenz Equations, Robbins Equations)


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
                                  -(x[3] - 3*x[2])*(1/tau_H)])"""

# Rikitake dynamo model for geomagnetic reversal, strange attractor

class RD(Model):
    def __init__(self, name = 'Rikitake Dynamo', x0=np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t_array = np.arange(0, 100, 0.01)):
        self.name = name
        self.x0 = x0
        self.t_array = t_array

    def eqns(self, x,t, m = 0.5, g = 50, r = 8, f = 0.5):
        return np.array([r*(x[3] - x[0]),
                         r*(x[2] - x[1]),
                         x[0]*x[4] + m*x[1] - (1 + m)*x[2],
                         x[1]*x[5] + m*x[0] - (1 + m)*x[3],
                         g*(1 - (1 + m)*x[0]*x[2] + m*x[0]*x[1]) - f*x[4],
                         g*(1 - (1 + m)*x[1]*x[3] + m*x[1]*x[0]) - f*x[5]])

# Lorenz atmospheric model, strange attractor

class L(Model):
    def __init__(self, name = 'Lorenz Equations', x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t_array = np.arange(0, 100, 0.01)):
        self.name = name
        self.x0 = x0
        self.t_array = t_array

    def eqns(self, x, t, sigma = 10.0, rho = 28.0, beta = 10.0/3):
        return np.array([sigma * (x[1] - x[0]),
                         rho*x[0] - x[1] - x[0]*x[2],
                         x[0]*x[1] - beta*x[2]])

# Robbins geomagnetic polarity reversal model

class R(Model):
    def __init__(self, name = 'Robbins Equations', x0 = np.array([0.00032,0.23,0.51]), t_array = np.arange(0, 200, 0.1)):
        self.name = name
        self.x0 = x0
        self.t_array = t_array

    def eqns(self,x,t, V = 1, sigma = 5, R = 13):
        return np.array([R - x[1]*x[2] - V*x[0],
                         x[0]*x[2] - x[1],
                         sigma*(x[1] - x[2])])