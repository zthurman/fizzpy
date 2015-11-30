#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzUnitTests
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import time as tm

# Model Test

def ModelTest():
    for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
        test = i
        print(test.name, test.xaxis, test.yaxis, test.x0, test.t0, test.t1, test.dt, test.t_array, test.eqns)

starttime = tm.time()
modeltester = ModelTest()
elapsedtime = (tm.time() - starttime)
print(" %s seconds" % elapsedtime)

# Solver Test

def SolverTest():
    for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
        test = i
        print(test.name, test.xaxis, test.yaxis, test.x0, test.t0, test.t1, test.dt, test.t_array, test.eqns)
        for j in [euler(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.eqns),
                  ord2(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.eqns),
                  rk4(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.eqns)]:
            soln = j
            print(soln.tsp, soln.Nsize, soln.X, soln.X[0], soln.X[:,0], soln.evaluate())

starttime1 = tm.time()
solvertester = SolverTest()
elapsedtime1 = (tm.time() - starttime1)


print("ModelTest took {0} seconds to execute while SolverTest took {1} seconds to execute" .format(elapsedtime, elapsedtime1))
