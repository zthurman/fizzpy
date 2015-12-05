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
        if type(test.name) != str:
            print 'Name attribute is not a string!'
        elif type(test.xaxis) != str:
            print 'Xaxis attribute is not a string!'
        elif type(test.yaxis) != str:
            print 'Yaxis attribute is not a string!'
        elif isinstance(test.x0, list):
            if isinstance(test.x0, float):
                if isinstance(test.x0, int):
                    print 'Initial conditions are not a list, float or integer!'
        elif isinstance(test.t0, float):
            if isinstance(test.t0, int):
                print 'Initial time is not a float or an integer!'
        elif isinstance(test.t1, float):
            if isinstance(test.t1, int):
                print 'Final time is not an float or an integer!'
        elif isinstance(test.dt, float):
            if isinstance(test.dt, int):
                print 'The timestep is not an integer or a float!'
        elif isinstance(test.t_array, list):
            print 'The time array is not a list!'
        else:
            print(test.name, test.xaxis, test.yaxis, test.x0, test.t0, test.t1, test.dt, test.t_array, test.equations)

starttime = tm.time()
modeltester = ModelTest()
elapsedtime = (tm.time() - starttime)

# Solver Test

def SolverTest():
    for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
        test = i
        print(test.name, test.xaxis, test.yaxis, test.x0, test.t0, test.t1, test.dt, test.t_array, test.equations)
        for j in [euler(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.equations),
                  ord2(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.equations),
                  rk4(test.name, test.xaxis, test.yaxis, test.x0, test.dt, test.t_array, test.equations)]:
            soln = j
            if isinstance(soln.tsp, list):
                print 'Tsp attribute is not a list!'
            if isinstance(soln.N_size, float):
                if isinstance(soln.N_size, int):
                    print 'Nsize is not a float or an integer!'
            if isinstance(soln.X, list):
                print 'X is not a list!'
            if isinstance(soln.X[0], float):
                if isinstance(soln.X[0], int):
                    print 'X[0] is not a float or an integer!'
            if isinstance(soln.X[:,0], list):
                print 'X[:,0] is not a list!'
            if isinstance(soln.evaluate(), list):
                print 'Solution array is not a list!'
            else:
                print(soln.tsp, soln.N_size, soln.X, soln.X[0], soln.X[:, 0], soln.evaluate())

starttime1 = tm.time()
solvertester = SolverTest()
elapsedtime1 = (tm.time() - starttime1)

print("ModelTest took {0} seconds to execute while SolverTest took {1} seconds to execute" .format(elapsedtime, elapsedtime1))
