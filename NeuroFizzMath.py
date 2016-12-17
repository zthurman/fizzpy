#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzMath
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv2


from __future__ import division
import numpy as np
import math as mt
import time as tm


def modelSelector(modelname):
    if modelname is None:
        raise TypeError
    if modelname == 'VDP':
        return 1
    elif modelname == 'LIF':
        return 2
    elif modelname == 'FN':
        return 3
    elif modelname == 'ML':
        return 4
    elif modelname == 'IZ':
        return 5
    elif modelname == 'HR':
        return 6
    elif modelname == 'RB':
        return 7
    elif modelname == 'HH':
        return 8
    elif modelname == 'RI':
        return 9
    else:
        return 0


def solverSelector(solvername):
    if solvername is None:
        raise TypeError
    if solvername == 'euler':
        return 1
    elif solvername == 'ord2':
        return 2
    elif solvername == 'rk4':
        return 3
    else:
        return 0


def modelMapper(modelname):
    if modelname in ['VDP', 'LIF', 'FN', 'ML', 'IZ', 'HR', 'RB', 'HH', 'RI']:
        if modelname == 'VDP':
            fullmodelname = VanDerPol
            return fullmodelname
        elif modelname == 'LIF':
            fullmodelname = LeakyIntegrateandFire
            return fullmodelname
        elif modelname == 'FN':
            fullmodelname = FitzhughNagumo
            return fullmodelname
        elif modelname == 'ML':
            fullmodelname = MorrisLecar
            return fullmodelname
        elif modelname == 'IZ':
            fullmodelname = Izhikevich
            return fullmodelname
        elif modelname == 'HR':
            fullmodelname = HindmarshRose
            return fullmodelname
        elif modelname == 'RB':
            fullmodelname = Robbins
            return fullmodelname
        elif modelname == 'HH':
            fullmodelname = HodgkinHuxley
            return fullmodelname
        elif modelname == 'RI':
            fullmodelname = Rikitake
            return fullmodelname
        else:
            return 'modelMapper is borked!'


def solverMapper(solvername):
    if solvername in ['euler', 'ord2', 'rk4']:
        if solvername == 'euler':
            fullsolvername = Euler
            return fullsolvername
        elif solvername == 'ord2':
            fullsolvername = SecondOrder
            return fullsolvername
        elif solvername == 'rk4':
            fullsolvername = RungeKutte4
            return fullsolvername
        else:
            return 'solverMapper is borked!'


def Euler(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        X[i + 1] = X[i] + k1 * dt
    return X, tsp


def SecondOrder(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(0, nsize - 1):
        k1 = model(X[i], tsp[i])
        k2 = model(X[i], tsp[i]) + k1 * (dt / 2)
        X[i + 1] = X[i] + k2 * dt
    return X, tsp


def RungeKutte4(t0=0, x0=np.array([1]), t1=5, dt=0.01, model=None):
    tsp = np.arange(t0, t1, dt)
    nsize = np.size(tsp)
    X = np.empty((nsize, np.size(x0)))
    X[0] = x0
    for i in range(1, nsize - 1):
        k1 = model(X[i], tsp[i])
        k2 = model(X[i] + dt/2*k1, tsp[i] + dt/2)
        k3 = model(X[i] + dt/2*k2, tsp[i] + dt/2)
        k4 = model(X[i] + dt*k3, tsp[i] + dt)
        X[i+1] = X[i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X, tsp


def VanDerPol(x, t):
    return np.array([x[1],
                    -x[0] + x[1]*(1-x[0]**2)])


def LeakyIntegrateandFire(x, t, u_th=-55, u_reset=-75, u_eq=-65, r=10, i=1.2):
    if x[0] >= u_th:
        x[0] = u_reset
    return np.array([-(x[0] - u_eq) + r*i])


def FitzhughNagumo(x, t, a=0.75, b=0.8, c=3, i=-0.40):
    return np.array([c*(x[0] + x[1] - x[0]**3/3 + i),
                    -1/c*(x[0] - a + b*x[1])])


def MorrisLecar(x, t, c=20, vk=-84, gk=8, vca=130, gca=4.4, vl=-60, gl=2, phi=0.04, v1=-1.2, v2=18, v3=2, v4=30,
                iapp=80):
    return np.array([(-gca*(0.5*(1 + mt.tanh((x[0] - v1)/v2)))*(x[0]-vca) - gk*x[1]*(x[0]-vk) - gl*(x[0]-vl) + iapp),
                    (phi*((0.5*(1 + mt.tanh((x[0] - v3)/v4))) - x[1]))/(1/mt.cosh((x[0] - v3)/(2*v4)))])


def Izhikevich(x, t, a=0.02, b=0.2, c=-65, d=2, i=10):
    if x[0] >= 30:
        x[0] = c
        x[1] = x[1] + d
    return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + i,
                    a*(b*x[0] - x[1])])


def HindmarshRose(x, t, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, i=1.3, xnot=-1.5):
    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + i,
                    c - d*(x[0]**2) - x[1],
                    r*(s*(x[0] - xnot) - x[2])])


def Robbins(x, t, V=1, sigma=5, R=13):
    return np.array([R - x[1]*x[2] - V*x[0],
                    x[0]*x[2] - x[1],
                    sigma*(x[1] - x[2])])


def HodgkinHuxley(x, t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
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


def Rikitake(x, t, m=0.5, g=50, r=8, f=0.5):
    return np.array([r*(x[3] - x[0]),
                     r*(x[2] - x[1]),
                     x[0]*x[4] + m*x[1] - (1 + m)*x[2],
                     x[1]*x[5] + m*x[0] - (1 + m)*x[3],
                     g*(1 - (1 + m)*x[0]*x[2] + m*x[0]*x[1]) - f*x[4],
                     g*(1 - (1 + m)*x[1]*x[3] + m*x[1]*x[0]) - f*x[5]])


def solutionGenerator(modelname, solvername):
    newmodelname = modelMapper(modelname)
    newsolvername = solverMapper(solvername)
    if modelSelector(modelname) in [1, 2, 3, 4, 5]:
        solution = newsolvername(x0=np.array([0.01, 0.01]), t1=100, dt=0.02, model=newmodelname)
        return solution[0], solution[1]
    elif modelSelector(modelname) in [6, 7]:
        solution = newsolvername(x0=np.array([0.01, 0.01, 0.01]), t1=100, dt=0.02, model=newmodelname)
        return solution[0], solution[1]
    elif modelSelector(modelname) == 8:
        solution = newsolvername(x0=np.array([0.01, 0.01, 0.01, 0.01]), t1=100, dt=0.02, model=newmodelname)
        return solution[0], solution[1]
    elif modelSelector(modelname) == 9:
        solution = newsolvername(x0=np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1=100, dt=0.0001, model=newmodelname)
        return solution[0], solution[1]
    else:
        solution = "Cheese please!"
        return solution


if __name__ == '__main__':
    startTime = tm.time()
    solutionArray = solutionGenerator('HH', 'euler')
    endTime = tm.time()
    elapsedTime = (endTime - startTime)
    print(solutionArray)
    print('The solver took ' + str(elapsedTime) + ' seconds to execute. Which is faster than '
                                                  'I could do it on paper so we\'ll call it good.')
