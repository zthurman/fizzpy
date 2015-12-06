#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzTimer
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, LIF, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt
import numpy as np
import time as tm

# Time execution of the different models evaluated against different solvers

def model_timer():
    time_array = []
    times = []
    for i in [VDP(), LIF(), FN(), ML(), IZ(), HR(), HH()]:
        model = i
        for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)]:
            starttime = tm.time()
            solvedmodel = j
            dynamicalvariable = solvedmodel.evaluate()
            endtime = tm.time()
            delta_t = (endtime - starttime)
            modelno = []
            solverno = []
            if solvedmodel.model_name == 'van der Pol oscillator':
                modelno = 1
            elif solvedmodel.model_name == 'Leaky integrate-and-fire':
                modelno = 2
            elif solvedmodel.model_name == 'Fitzhugh-Nagumo':
                modelno = 3
            elif solvedmodel.model_name == 'Morris-Lecar':
                modelno = 4
            elif solvedmodel.model_name == 'Izhikevich':
                modelno = 5
            elif solvedmodel.model_name == 'Hindmarsh-Rose':
                modelno = 6
            elif solvedmodel.model_name == 'Hodgkin-Huxley':
                modelno = 7
            else:
                modelno = 0

            if solvedmodel.name == 'Euler':
                solverno = 1
            elif solvedmodel.name == 'Second Order Runge-Kutta':
                solverno = 2
            elif solvedmodel.name == 'Fourth Order Runge-Kutta':
                solverno = 3
            else:
                solverno = 0

            times += [(delta_t, modelno, solverno)]
        time_array = times
    return time_array

timed = model_timer()
print len(timed)
# print timed
print timed[0]
vdp = timed[0]
print vdp[0]

# Create a plot for the time to evaluate each model-solver combination

# def do_timeplot():
#     timed = model_timer()
#     for i in np.arange(0, 20):
#         model = timed[i]
#         for j in np.arange(0, 2):
#             plt.figure()
#             plt.plot()
