#!/usr/bin/env python
#  Eventually I'd like to create an application that provides a nice intuitive GUI that enables the user to #manipulate different models of differential equations to exhibit different behaviors given different input #parameters.

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt

# First we need to come up with a class that will be general enough to encompass all of the neuron models
# Step 1: Create the Neuron class to be the most general, all specific neuron models should inherit from this class
# Things to determine, what is common to all of the models? parameters and equations
#	-The parameters are constants fed into the equations when they are solved, will not need to be defined until
#        the system is evaluated
#	-The equations consist of a combination of dynamical variables and constants which will need to be evaluated
# 	 together to come up with the solution

class Neuron:
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

	def FN(x,t,a = 0.75,b = 0.8,c = 3, I = -0.40):
    		return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I), \
                    	-1/c*(x[0]- a + b*x[1])])
	
	def ML(v,t,c = 20,vk = -84,gk = 8,vca = 120,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 90):
    		return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])
	
	def HR(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 2.5, xnot = -1.5):
    		return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I, \
                        c - d*(x[0]**2) - x[1], \
                        r*(s*(x[0] - xnot) - x[2])])


Q = Neuron()
	
def pplot(Q):
	X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = Q)
	
def tplot(Q):
	X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = Q)

def fftplot(Q):
	X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = Q)
	
def p2plot(Q):
	X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = Q)
	
def t2plot(Q):
	X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = Q)

