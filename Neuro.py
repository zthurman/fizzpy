#!/usr/bin/env python

#  This is a library of functions that allow for numerical solutions to systems of 
# nonlinear differential equations that govern individual neuron behavior to be
# found.

# Imports
from __future__ import division
from scipy import *
from numpy import *
import numpy as np
import pylab
import matplotlib as mp
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft, fftfreq

#  RK4 - 4th order Runge-Kutte algorithm, used as an alternative to the built-in
# python ODE solvers because the built-in solvers have a variable time step. 
# For neural systems where the voltage varies rapidly very much like the Dirac
# delta function experience has shown that variable time step solvers are often
# not well suited for such applications.

def RK4(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):  
    tsp = np.arange(t0, t1, dt)
    Nsize = np.size(tsp)
    X = np.empty((Nsize, np.size(x0)))
    X[0] = x0

    for i in range(1, Nsize):
        k1 = ng(X[i-1],tsp[i-1])
        k2 = ng(X[i-1] + dt/2*k1, tsp[i-1] + dt/2)
        k3 = ng(X[i-1] + dt/2*k2, tsp[i-1] + dt/2)
        k4 = ng(X[i-1] + dt*k3, tsp[i-1] + dt)
        X[i] = X[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X

# Fitzhugh-Nagumo

#  FN - Defining the Fitzhugh-Nagumo system of x(dot) and r(dot)
# inputs a, b and c give biological behavior, I is the input stimulus
# x is the membrane potential variable, r is the membrane recovery variable

def FN(x,t,a = 0.75,b = 0.8,c = 3, I = -1.476):
    return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I), \
                    -1/c*(x[0]- a + b*x[1])])

#  Different plotting functions for the FN function that utilize RK4

# Phase plot of membrane potential variable against membrane recovery
# variable.

def do_FNpplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    pylab.plot(X[:,1], X[:,0])
    pylab.title("Phase Portrait")
    pylab.xlabel("Membrane Recovery Variable")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('FNpplot.png')
    pylab.show()
    return

# Time plot of the membrane potential variable of the system over time

def do_FNtplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled FN neuron")
    pylab.xlabel("Time")
    pylab.savefig('FNtplot.png')
    pylab.show()
    return

# Fourier transform of the membrane potential signal

def do_FNfftplot():
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[4:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[4:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal - FN")
    pylab.xlabel("Frequency (kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,0.4)
    pylab.ylim(0,2e7)
    pylab.savefig('FNfftplot.png')
    pylab.show()
    return

#  Now for two coupled neurons

#  Create a function for two linearly coupled neurons with a coupling constant of k
# second neuron has no input stimulus because it's coupled to the first
# by varying the coupling constant can moderate the behavior of both neurons

def FN2(x,t,a = 0.75,b = 0.8,c = 3, I = -0.80, k = 0.75):
    return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I + k*(x[2] - x[0])), \
                    -1/c*(x[0]- a + b*x[1]), \
                     c*(x[2]+ x[3]- x[2]**3/3 + k*(x[0] - x[2])),
                    -1/c*(x[2]- a + b*x[3])])

# Phase plot of membrane potential variable against membrane recovery
# variable for two linearly coupled neurons.

def do_FNp2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = FN2)
    pylab.plot(X[:,1], X[:,0])
    pylab.plot(X[:,3], X[:,2])
    pylab.title("Phase Portrait - linearly coupled FN")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('FNp2plot.png')
    pylab.show()
    return

# Time plot of the membrane potential variable of the system over time

def do_FNt2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = FN2)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.plot(tsp,X[:,2])
    pylab.title("Membrane Potential over Time - two linearly coupled FN neurons")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('FNt2plot.png')
    pylab.show()
    return

# Morris-Lecar

# Super-critical Hopf bifurcation for the system somewhere in the region of Iapp = 79.352-79.353 with Iapp constant
# Lots of parameters, look them up for more details on why they're chosen as such.

def ML(v,t,c = 20,vk = -84,gk = 8,vca = 130,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 79):
    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])

def ML2(v,t,c = 20,vk = -84,gk = 8,vca = 120,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 125,k = 1.5):
    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp + k*(v[2] - v[0])), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4))),
                     (-gca*(0.5*(1 + mt.tanh((v[2] - v1)/v2)))*(v[2]-vca) - gk*v[3]*(v[2]-vk) - gl*(v[2]-vl) + k*(v[0] - v[2])), 
                     (phi*((0.5*(1 + mt.tanh((v[2] - v3)/v4))) - v[3]))/(1/mt.cosh((v[2] - v3)/(2*v4)))])  

def do_MLpplot():
    pylab.figure()
    X = RK4(x0 = np.array([0,0]), t1 = 1000,dt = 0.1, ng = ML)
    pylab.plot(X[:,0], X[:,1])
    pylab.title("Phase Portrait - single uncoupled ML neuron")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('MLpplot.png')
    pylab.show()
    return

def do_MLtplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 1000,dt = 0.1, ng = ML)
    t0 = 0
    t1 = 1000
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('MLtplot.png')
    pylab.show()
    return

# Compute the Power spectrum of the neuron membrane potential over time

def do_MLfftplot():
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 800,dt = 0.1, ng = ML)
    Y = mean(X)		# determine DC component of signal
    X = X - Y		# subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(X.size/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency (kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,0.4)
    pylab.ylim(0,2e10)
    pylab.savefig('MLfftplot.png')
    pylab.show()
    print X.size
    return

# Izhikevich

# Parameter ranges: I = 10
# ~regular spiking: a = 0.02, b = 0.2, c = -65, d = 2
# ~fast spiking: a = 0.1, b = 0.2, c = -65, d = 2
# ~bursting: a = 0.02, b = 0.2, c = -50, d = 2


# Supercritical Hopf with fast spiking parameters between: I = 3.77437 - 3.77438

def Izhi(x,t, a = 0.02, b = 0.2, c = -65, d = 2, I = 10):
    if x[0] >= 30:
        x[0] = c
        x[1] = x[1] + d
    return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + I, \
                    a*(b*x[0] - x[1])])

def do_Izhipplot():
    pylab.figure()
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    pylab.plot(X[:,1], X[:,0])
    pylab.title("Phase Portrait - Izhikevich")
    pylab.xlabel("Membrane Recovery Variable")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('Izhipplot.png')
    pylab.show()
    return

def do_Izhitplot():
    pylab.figure()
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    t0 = 0
    t1 = 300
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - Izhikevich")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential (mV)")
    pylab.ylim(-80,40)
    pylab.savefig('Izhitplot.png')
    pylab.show()
    return

def do_Izhifftplot():
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal - Izhikevich")
    pylab.xlabel("Frequency")
    pylab.ylabel("Power")
    pylab.xlim(0,0.6)
    pylab.ylim(0,1.75e10)
    pylab.savefig('Izhifftplot.png')
    pylab.show()
    return

# Hindmarsh-Rose

def HR(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 1.3, xnot = -1.5):
    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I, \
                        c - d*(x[0]**2) - x[1], \
                        r*(s*(x[0] - xnot) - x[2])])

def HR2(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 1.84, xnot = -1.5, k = 0.05):
    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I + k*(x[3] - x[0]), \
                    c - d*(x[0]**2) - x[1], \
                    r*(s*(x[0] - xnot) - x[2]), \
                    x[4] - a*(x[3]**3) + (b*(x[3]**2)) - x[5] + I + k*(x[0] - x[3]), \
                    c - d*(x[3]**2) - x[4], \
                    r*(s*(x[3] - xnot) - x[5])])

def do_HRpplot():
    pylab.figure()
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    pylab.plot(X[:,0], X[:,1])
    pylab.title("Phase Portrait - Hindmarsh-Rose")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('HRpplot.png')
    pylab.show()
    return

def do_HRtplot():
    pylab.figure()
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled HR neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('HRtplot.png')
    pylab.show()
    return

def do_HRfftplot():
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    Y = mean(X[:,0])
    X[:,0] = X[:,0] - Y
    fdata = X[:,0].size
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency ~(kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,1)
    pylab.ylim(0,4e6)
    pylab.show()
    return

def do_HRp2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01]), t1 = 600,dt = 0.01, ng = HR2)
    pylab.plot(X[:,1], X[:,0])
    pylab.plot(X[:,4], X[:,3])
    pylab.title("Phase Portrait - linearly coupled HR")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('HRp2plot.png')
    pylab.show()
    return

def do_HRt2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01]), t1 = 600,dt = 0.01, ng = HR2)
    t0 = 0
    t1 = 600
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.plot(tsp,X[:,3])
    pylab.title("Membrane Potential over Time - linearly coupled HR")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential (mV)")
    pylab.xlim(100,600)
    pylab.savefig('HRt2plot.png')
    pylab.show()
    return

# Hodgkins-Huxley

def HH(x,t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
    alpha_n = (0.01*(x[0]+10))/(exp((x[0]+10)/10)-1)
    beta_n = 0.125*exp(x[0]/80)
    alpha_m = (0.1*(x[0]+25))/(exp((x[0]+25)/10)-1)
    beta_m = 4*exp(x[0]/18)
    alpha_h = (0.07*exp(x[0]/20))
    beta_h = 1 / (exp((x[0]+30)/10)+1)
    return np.array([(g_K*(x[1]**4)*(x[0]-E_K) + g_Na*(x[2]**3)*x[3]*(x[0]-E_Na) + g_L*(x[0]-E_L) - I)*(-1/C_m), \
                      alpha_n*(1-x[1]) - beta_n*x[1], \
                      alpha_m*(1-x[2]) - beta_m*x[2], \
                      alpha_h*(1-x[3]) - beta_h*x[3]])

def do_HHpplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.01, ng = HH)
    pylab.plot(X[:,1], X[:,0])
    pylab.title("Phase Portrait - HH")
    pylab.xlabel("Potassium gating variable")
    pylab.ylabel("Membrane Potential")
    pylab.savefig("HHpplot.png")
    pylab.show()
    return

def do_HHtplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.01, ng = HH)
    t0 = 0
    t1 = 100
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,-X[:,0])
    pylab.title("Membrane Potential over Time - HH")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig("HHtplot.png")
    #pylab.xlim(0,400)
    #pylab.ylim(-5,35)
    pylab.show()
    return

def do_HHfftplot():
    X = RK4(x0 = np.array([0,0,0,0]), t1 = 100,dt = 0.01, ng = HH)
    Y = mean(X[:,0])
    X[:,0] = X[:,0] - Y
    fdata = X[:,0].size
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency ~(kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,1)
    pylab.ylim(0,1e10)
    pylab.savefig('HHfftplot.png')
    pylab.show()
    return
