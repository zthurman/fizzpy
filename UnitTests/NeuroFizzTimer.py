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
    for i in [VDP(), LIF(), FN(), ML(), IZ(), HR(), HH()]:
        model = i
        time_array = []
        for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)]:
            starttime = tm.time()
            solvedmodel = j
            dynamicalvariable = solvedmodel.evaluate()
            endtime = tm.time()
            delta_t = (endtime - starttime)
            # times = "{0} seconds for {1} against {2}" .format(delta_t, solvedmodel.model_name, solvedmodel.name)
            times = ([delta_t, solvedmodel.model_name, solvedmodel.name])
            time_array.append(times)

    return time_array

timed = model_timer()
print len(timed)
print timed


# times = []
# for i in [VDP(), LIF(), FN(), ML(), IZ(), HR(), HH()]:
#     model = i
#     times = []
#     for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
#               ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
#               rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)]:
#         starttime = tm.time()
#         solvedmodel = j
#         dynamicalvariable = solvedmodel.evaluate()
#         endtime = tm.time()
#         delta_t = (endtime - starttime)
#         # times = "{0} seconds for {1} against {2}" .format((endtime - starttime), solvedmodel.model_name, solvedmodel.name)
#         times.append([delta_t, solvedmodel.model_name, solvedmodel.name])
#     print times