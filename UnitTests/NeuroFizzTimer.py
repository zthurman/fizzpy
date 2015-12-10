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
            times += [(delta_t, solvedmodel.model_name, solvedmodel.name)]
        time_array = times
    return time_array

timed = model_timer()
print len(timed)
# print timed
print timed
vdp = timed[0]
print vdp[0]

# Create a plot for the time to evaluate each model-solver combination

# def do_timeplot():
#     timed = model_timer()
#     times = []
#     for i in np.arange(0, 20):
#         model = timed[i]
#         times = model[0]
#         print times

# Euler plots

timed = model_timer()
times = []
my_xticks = []
for i in np.arange(0, 20, 3):
    model = timed[i]
    times += [(model[0])]
    # my_xticks += [model[1]]

plt.figure()
# plt.xticks(my_xticks)
plt.plot(times, '.')
plt.savefig('euler.png')

# Second Order Plots

times = []
for i in np.arange(1, 20, 3):
    model = timed[i]
    times += [(model[0])]

plt.figure()
plt.plot(times, '.')
plt.savefig('ord2.png')

# Fourth Order Plots

times = []
for i in np.arange(2, 21, 3):
    model = timed[i]
    times += [(model[0])]

plt.figure()
plt.plot(times, '.')
plt.savefig('rk4.png')

