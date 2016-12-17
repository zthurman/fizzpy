#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzMath
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from numpy import array, arange, size, empty
from time import time
from math import exp, tanh, cosh


# Dictionary of available models with solution array dimension

modelDictionary = {'LIF': 1, 'VDP': 2, 'SHM': 2, 'FN': 2, 'ML': 2, 'IZ': 2, 'HR': 3, 'RB': 3, 'HH': 4, 'RI': 6}


# Solver functions

def Euler(t0=0, x0=array([1]), t1=5, dt=0.01, model=None):
    tsp = arange(t0, t1, dt)
    nsize = size(tsp)
    X = empty((nsize, size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        X[i + 1] = X[i] + k1 * dt
    return X, tsp


def SecondOrder(t0=0, x0=array([1]), t1=5, dt=0.01, model=None):
    tsp = arange(t0, t1, dt)
    nsize = size(tsp)
    X = empty((nsize, size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        k2 = model(X[i], tsp[i]) + k1 * (dt / 2)
        X[i + 1] = X[i] + k2 * dt
    return X, tsp


def RungeKutta4(t0=0, x0=array([1]), t1=5, dt=0.01, model=None):
    tsp = arange(t0, t1, dt)
    nsize = size(tsp)
    X = empty((nsize, size(x0)))
    X[0] = x0
    for i in range(1, nsize - 1):
        k1 = model(X[i], tsp[i])
        k2 = model(X[i] + dt/2*k1, tsp[i] + dt/2)
        k3 = model(X[i] + dt/2*k2, tsp[i] + dt/2)
        k4 = model(X[i] + dt*k3, tsp[i] + dt)
        X[i+1] = X[i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X, tsp


# Model functions

def LeakyIntegrateandFire(x, t, u_th=-55, u_reset=-75, u_eq=-65, r=10, i=1.2):
    if x[0] >= u_th:
        x[0] = u_reset
    return array([-(x[0] - u_eq) + r*i])


def VanDerPol(x, t):
    return array([x[1],
                 -x[0] + x[1]*(1-x[0]**2)])


def DampedSHM(x, t, r=0.035, s=0.5, m=0.2):
    return array([x[1],
                 (-r*x[1] - s*x[0])/m])


def FitzhughNagumo(x, t, a=0.75, b=0.8, c=3, i=-0.40):
    return array([c*(x[0] + x[1] - x[0]**3/3 + i),
                 -1/c*(x[0] - a + b*x[1])])


def MorrisLecar(x, t, c=20, vk=-84, gk=8, vca=130, gca=4.4, vl=-60, gl=2, phi=0.04, v1=-1.2, v2=18, v3=2, v4=30,
                iapp=80):
    return array([(-gca*(0.5*(1 + tanh((x[0] - v1)/v2)))*(x[0]-vca) - gk*x[1]*(x[0]-vk) - gl*(x[0]-vl) + iapp),
                 (phi*((0.5*(1 + tanh((x[0] - v3)/v4))) - x[1]))/(1/cosh((x[0] - v3)/(2*v4)))])


def Izhikevich(x, t, a=0.02, b=0.2, c=-65, d=2, i=10):
    if x[0] >= 30:
        x[0] = c
        x[1] = x[1] + d
    return array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + i,
                 a*(b*x[0] - x[1])])


def HindmarshRose(x, t, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, i=1.3, xnot=-1.5):
    return array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + i,
                 c - d*(x[0]**2) - x[1],
                 r*(s*(x[0] - xnot) - x[2])])


def Robbins(x, t, V=1, sigma=5, R=13):
    return array([R - x[1]*x[2] - V*x[0],
                 x[0]*x[2] - x[1],
                 sigma*(x[1] - x[2])])


def HodgkinHuxley(x, t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
    alpha_n = (0.01*(x[0]+10))/(exp((x[0]+10)/10)-1)
    beta_n = 0.125*exp(x[0]/80)
    alpha_m = (0.1*(x[0]+25))/(exp((x[0]+25)/10)-1)
    beta_m = 4*exp(x[0]/18)
    alpha_h = (0.07*exp(x[0]/20))
    beta_h = 1 / (exp((x[0]+30)/10)+1)
    return array([(g_K*(x[1]**4)*(x[0]-E_K) + g_Na*(x[2]**3)*x[3]*(x[0]-E_Na) + g_L*(x[0]-E_L) - I)*(-1/C_m),
                 alpha_n*(1-x[1]) - beta_n*x[1],
                 alpha_m*(1-x[2]) - beta_m*x[2],
                 alpha_h*(1-x[3]) - beta_h*x[3]])


def Rikitake(x, t, m=0.5, g=50, r=8, f=0.5):
    return array([r*(x[3] - x[0]),
                 r*(x[2] - x[1]),
                 x[0]*x[4] + m*x[1] - (1 + m)*x[2],
                 x[1]*x[5] + m*x[0] - (1 + m)*x[3],
                 g*(1 - (1 + m)*x[0]*x[2] + m*x[0]*x[1]) - f*x[4],
                 g*(1 - (1 + m)*x[1]*x[3] + m*x[1]*x[0]) - f*x[5]])


# Housekeeping functions

def getModelDictionaryKeys(modeldictionary):
    modeldictionarykeys = sorted(modeldictionary.keys())
    return modeldictionarykeys


def getModelDictionaryValues(modeldictionary):
    modeldictionaryvalues = sorted(modeldictionary.values())
    return modeldictionaryvalues


def getModelDictionaryValue(modelkey):
    modeldictionaryvalue = modelDictionary.get(modelkey)
    return modeldictionaryvalue


def modelSelector(modelname):
    if modelname is None:
        raise TypeError
    if modelname == 'LIF':
        return LeakyIntegrateandFire
    elif modelname == 'VDP':
        return VanDerPol
    elif modelname == 'SHM':
        return DampedSHM
    elif modelname == 'FN':
        return FitzhughNagumo
    elif modelname == 'ML':
        return MorrisLecar
    elif modelname == 'IZ':
        return Izhikevich
    elif modelname == 'HR':
        return HindmarshRose
    elif modelname == 'RB':
        return Robbins
    elif modelname == 'HH':
        return HodgkinHuxley
    elif modelname == 'RI':
        return Rikitake
    else:
        return 1


def solverSelector(solvername):
    if solvername is None:
        raise TypeError
    if solvername == 'euler':
        return Euler
    elif solvername == 'ord2':
        return SecondOrder
    elif solvername == 'rk4':
        return RungeKutta4
    else:
        return 1


def dimensionIdentifier(modelname):
    if modelname is None:
        raise TypeError
    if modelname in getModelDictionaryKeys(modelDictionary):
        dimension = getModelDictionaryValue(modelname)
        return dimension
    else:
        return 1


def distinctdimensionIdentifier():
    availabledimensions = set(modelDictionary.values())
    return availabledimensions


def initIdentifier(modelname):
    if modelname in getModelDictionaryKeys(modelDictionary):
        dimension = dimensionIdentifier(modelname)
        if dimension == 1:
            x0 = array([0.01])
            return x0
        elif dimension == 2:
            x0 = array([0.01, 0.01])
            return x0
        elif dimension == 3:
            x0 = array([0.01, 0.01, 0.01])
            return x0
        elif dimension == 4:
            x0 = array([0.01, 0.01, 0.01, 0.01])
            return x0
        elif dimension == 6:
            x0 = array([-1.4, -1, -1, -1.4, 2.2, -1.5])
            return x0
    else:
        return 1


def endtimeIdentifier():
    t1 = 100
    return t1


def timestepIdentifier(modelname):
    if modelname != 'RI':
        dt = 0.02
        return dt
    elif modelname == 'RI':
        dt = 0.0001
        return dt
    else:
        return 1


# Workhorse function

def solutionGenerator(modelname, solvername):
    newmodelname = modelSelector(modelname)
    newsolvername = solverSelector(solvername)
    if modelname in getModelDictionaryKeys(modelDictionary):
        solution = newsolvername(x0=initIdentifier(modelname), t1=endtimeIdentifier(),
                                 dt=timestepIdentifier(modelname), model=newmodelname)
        return solution[0], solution[1]
    else:
        solution = "Dude...somethings BORKED!"
        return solution


# Main

if __name__ == '__main__':
    startTime = time()
    solutionArray = solutionGenerator('SHM', 'euler')
    endTime = time()
    elapsedTime = (endTime - startTime)
    print(solutionArray)
    print('The solver took ' + str(elapsedTime) + ' seconds to execute. Which is faster than '
                                                  'I could do it on paper so we\'ll call it good.')

