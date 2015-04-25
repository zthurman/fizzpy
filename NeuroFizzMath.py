#!/usr/bin/env python
#  Classes for different neuron models that all inherit from parent neuron.

from __future__ import division
from Neuro import *
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt

class Neuron:
	def __init__(self, params, inputs, model):
	self.params = []
	self.inputs = []
	
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

class FN(Neuron):
	self.params = {a = 0.75, b = 0.8, c = 3}
	self.inputs = -1.476
	def FN(x,t,self.params['a'], self.params['b'], self.params['c'], self.inputs):
	    return np.array([self.params['c']*(x[0]+ x[1]- x[0]**3/3 + self.inputs), \
		            -1/self.params['c']*(x[0]- self.params['a'] + self.params['b']*x[1])])
	  

class ML(Neuron):
	self.params = {}
	self.inputs = 
	def ML(v,t,c = 20,vk = -84,gk = 8,vca = 130,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 79):
	    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
		             (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])


class IZ(Neuron):
	self.params = {}
	self.inputs = 
	def Izhi(x,t, a = 0.02, b = 0.2, c = -65, d = 2, I = 10):
	    if x[0] >= 30:
		x[0] = c
		x[1] = x[1] + d
	    return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + I, \
			    a*(b*x[0] - x[1])])

class HR(Neuron):
	self.params = {}
	self.inputs = 
	def HR2(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 1.84, xnot = -1.5, k = 0.05):
	    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I + k*(x[3] - x[0]), \
		            c - d*(x[0]**2) - x[1], \
		            r*(s*(x[0] - xnot) - x[2]), \
		            x[4] - a*(x[3]**3) + (b*(x[3]**2)) - x[5] + I + k*(x[0] - x[3]), \
		            c - d*(x[3]**2) - x[4], \
		            r*(s*(x[3] - xnot) - x[5])])

class HH(Neuron):
	self.params = {}
	self.inputs = 
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
	
	
		
