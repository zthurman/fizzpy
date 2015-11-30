#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt
import time as tm

# Time execution of the VDP evaluate with different solvers

# Euler Solver

starttime = tm.time()
eqns = HH()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds" % (tm.time() - starttime))

# Second Order Runge-Kutte Solver

starttime = tm.time()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds" % (tm.time() - starttime))

# Fourth Order Runge-Kutte

starttime = tm.time()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds" % (tm.time() - starttime))


# Do a test plot and save it as a png in the working directory

def do_tplot():
    neuron = HH()
    solvedneuron = ord2(neuron.name, neuron.xaxis, neuron.yaxis, neuron.x0, neuron.dt, neuron.t_array, neuron.eqns)
    membranepotential = solvedneuron.evaluate()
    plt.figure()
    plt.plot(solvedneuron.tsp, -membranepotential[:, 0])
    plt.title(solvedneuron.modelname)
    plt.xlabel(solvedneuron.xaxis)
    plt.ylabel(solvedneuron.yaxis)
    plt.savefig('VDPtplot.png')
    plt.show()
    return

print do_tplot()
