#!/usr/bin/env python
# FizzPyX - FizzPyX
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from numpy import array, arange, size, empty
from time import time
from math import exp, tanh, cosh


# Dictionary of available models with solution array dimension

modelDictionary = {'LIF': 1, 'VDP': 2, 'SHM': 2, 'FN': 2, 'ML': 2, 'IZ': 2, 'HR': 3, 'RB': 3, 'LO': 3, 'HH': 4, 'CO': 4,
                   'RI': 6}


# List of available solvers

solverList = ['euler', 'ord2', 'rk4']


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


# Uncoupled Model functions

def LeakyIntegrateandFire(x, t, u_th=None, u_reset=None, u_eq=None, r=None, i=None):
    u_th = (-55, u_th)[u_th is not None]
    u_reset = (-75, u_reset)[u_reset is not None]
    u_eq = (-65, u_eq)[u_eq is not None]
    r = (10, r)[r is not None]
    i = (1.2, i)[i is not None]
    if x[0] >= u_th:
        x[0] = u_reset
    return array([-(x[0] - u_eq) + r*i])


def VanDerPol(x, t, mu=None):
    mu = (1, mu)[mu is not None]
    return array([x[1],
                 mu*x[1]*(1-x[0]**2)-x[0]])


def DampedSHM(x, t, r=None, s=None, m=None):
    r = (0.035, r)[r is not None]
    s = (0.5, s)[s is not None]
    m = (0.2, m)[m is not None]
    return array([x[1],
                 (-r*x[1] - s*x[0])/m])


def FitzhughNagumo(x, t, a=None, b=None, c=None, i=None):
    a = (0.75, a)[a is not None]
    b = (0.8, b)[b is not None]
    c = (3, c)[c is not None]
    i = (-0.4, i)[i is not None]
    return array([c*(x[0] + x[1] - x[0]**3/3 + i),
                 -1/c*(x[0] - a + b*x[1])])


def MorrisLecar(x, t, vk=None, gk=None, vca=None, gca=None, vl=None, gl=None, phi=None, v1=None, v2=None, v3=None,
                v4=None, iapp=None):
    vk = (-84, vk)[vk is not None]
    gk = (8, gk)[gk is not None]
    vca = (130, vca)[vca is not None]
    gca = (4.4, gca)[gca is not None]
    vl = (-60, vl)[vl is not None]
    gl = (2, gl)[gl is not None]
    phi = (0.04, phi)[phi is not None]
    v1 = (-1.2, v1)[v1 is not None]
    v2 = (18, v2)[v2 is not None]
    v3 = (2, v3)[v3 is not None]
    v4 = (30, v4)[v4 is not None]
    iapp = (80, iapp)[iapp is not None]
    return array([(-gca*(0.5*(1 + tanh((x[0] - v1)/v2)))*(x[0]-vca) - gk*x[1]*(x[0]-vk) - gl*(x[0]-vl) + iapp),
                 (phi*((0.5*(1 + tanh((x[0] - v3)/v4))) - x[1]))/(1/cosh((x[0] - v3)/(2*v4)))])


def Izhikevich(x, t, a=None, b=None, c=None, d=None, i=None):
    a = (0.02, a)[a is not None]
    b = (0.2, b)[b is not None]
    c = (-65, c)[c is not None]
    d = (2, d)[d is not None]
    i = (10, i)[i is not None]
    if x[0] >= 30:
        x[0] = c
        x[1] += d
    return array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + i,
                 a*(b*x[0] - x[1])])


def HindmarshRose(x, t, a=None, b=None, c=None, d=None, r=None, s=None, i=None, xnot=None):
    a = (1.0, a)[a is not None]
    b = (3.0, b)[b is not None]
    c = (1.0, c)[c is not None]
    d = (5.0, d)[d is not None]
    r = (0.006, r)[r is not None]
    s = (4.0, s)[s is not None]
    i = (1.3, i)[i is not None]
    xnot = (-1.5, xnot)[xnot is not None]
    return array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + i,
                 c - d*(x[0]**2) - x[1],
                 r*(s*(x[0] - xnot) - x[2])])


def Robbins(x, t, V=None, sigma=None, R=None):
    V = (1, V)[V is not None]
    sigma = (5, sigma)[sigma is not None]
    R = (13, R)[R is not None]
    return array([R - x[1]*x[2] - V*x[0],
                 x[0]*x[2] - x[1],
                 sigma*(x[1] - x[2])])


def Lorenz(x, t, sigma=None, rho=None, beta=None):
    sigma = (10.0, sigma)[sigma is not None]
    rho = (28.0, rho)[rho is not None]
    beta = (10.0/3, beta)[beta is not None]
    return array([sigma * (x[1] - x[0]),
                 rho*x[0] - x[1] - x[0]*x[2],
                 x[0]*x[1] - beta*x[2]])


def HodgkinHuxley(x, t, g_K=None, g_Na=None, g_L=None, E_K=None, E_Na=None, E_L=None, C_m=None, I=None):
    g_K = (36, g_K)[g_K is not None]
    g_Na = (120, g_Na)[g_Na is not None]
    g_L = (0.3, g_L)[g_L is not None]
    E_K = (12, E_K)[E_K is not None]
    E_Na = (-115, E_Na)[E_Na is not None]
    E_L = (-10.613, E_L)[E_L is not None]
    C_m = (1, C_m)[C_m is not None]
    I = (-10, I)[I is not None]
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


def Rikitake(x, t, m=None, g=None, r=None, f=None):
    m = (0.5, m)[m is not None]
    g = (50, g)[g is not None]
    r = (8, r)[r is not None]
    f = (0.5, f)[f is not None]
    return array([r*(x[3] - x[0]),
                 r*(x[2] - x[1]),
                 x[0]*x[4] + m*x[1] - (1 + m)*x[2],
                 x[1]*x[5] + m*x[0] - (1 + m)*x[3],
                 g*(1 - (1 + m)*x[0]*x[2] + m*x[0]*x[1]) - f*x[4],
                 g*(1 - (1 + m)*x[1]*x[3] + m*x[1]*x[0]) - f*x[5]])


# Coupled model functions

def CoupledOscillators(x, t, b=None, k1=None, k2=None, m=None):
    b = (0.01, b)[b is not None]
    k1 = (0.27, k1)[k1 is not None]
    k2 = (0.027, k2)[k2 is not None]
    m = (0.25, m)[m is not None]
    return array([x[1],
                 -(k1/m)*x[0] + (k2/m)*x[2] - (b/m)*x[1],
                 x[3],
                 (k2/m)*x[0] - (k1/m)*x[2] - (b/m)*x[3]])


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
    elif modelname == 'LO':
        return Lorenz
    elif modelname == 'HH':
        return HodgkinHuxley
    elif modelname == 'CO':
        return CoupledOscillators
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


def initIdentifier(modelname, inits=None):
    if modelname in getModelDictionaryKeys(modelDictionary):
        dimension = dimensionIdentifier(modelname)
        if inits is not None:
            x0 = inits
            return x0
        elif dimension == 1:
            x0 = array([0.01])
            return x0
        elif dimension == 2:
            x0 = array([0.01, 0.01])
            return x0
        elif dimension == 3:
            x0 = array([0.01, 0.01, 0.01])
            return x0
        elif dimension == 4 and modelname == 'CO':
            x0 = array([0, 0, 0.5, 0])
            return x0
        elif dimension == 4:
            x0 = array([0.01, 0.01, 0.01, 0.01])
            return x0
        elif dimension == 6:
            x0 = array([-1.4, -1, -1, -1.4, 2.2, -1.5])
            return x0
    else:
        return 1


def endtimeIdentifier(modelname, endtime=None):
    if modelname in getModelDictionaryKeys(modelDictionary):
        if endtime is not None:
            t1 = endtime
            return t1
        elif modelname == 'CO':
            t1 = 200
            return t1
        else:
            t1 = 100
            return t1


def timestepIdentifier(modelname, timestep=None):
    if timestep is not None:
        dt = timestep
        return dt
    elif modelname != 'RI':
        dt = 0.02
        return dt
    elif modelname == 'RI':
        dt = 0.0001
        return dt
    else:
        return 1


# Workhorse function

def solutionGenerator(modelname, solvername, inits=None, endtime=None, timestep=None):
    newmodelname = modelSelector(modelname)
    newsolvername = solverSelector(solvername)
    if modelname in getModelDictionaryKeys(modelDictionary) and ('inits' is not None or
                                                                 'endtime' is not None or
                                                                 'timestep' is not None):
        if inits is not None:
            inits = inits
        if endtime is not None:
            endtime = endtime
        if timestep is not None:
            timestep = timestep
        solution = newsolvername(t0=0, x0=initIdentifier(modelname, inits=inits),
                                 t1=endtimeIdentifier(modelname, endtime=endtime),
                                 dt=timestepIdentifier(modelname, timestep=timestep), model=newmodelname)
        return solution[0], solution[1]
    elif modelname in getModelDictionaryKeys(modelDictionary):
        solution = newsolvername(x0=initIdentifier(modelname), t1=endtimeIdentifier(modelname),
                                 dt=timestepIdentifier(modelname), model=newmodelname)
        return solution[0], solution[1]
    else:
        solution = "Dude...somethings BORKED!"
        return solution


# Main

if __name__ == '__main__':
    startTime = time()

    # Execution example for default parameters: x0=array([0.01, 0.01]), t1=100, dt=0.02
    # solutionArray = solutionGenerator('CO', 'ord2')

    # Execution example for non-default parameters:
    solutionArray = solutionGenerator('ML', 'euler')

    endTime = time()
    elapsedTime = (endTime - startTime)
    print(solutionArray)
    print('The solver took ' + str(elapsedTime) + ' seconds to execute. Which is faster than '
                                                  'I could do it on paper so we\'ll call it good.')
