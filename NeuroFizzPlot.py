#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt

# Do some test plots and save outputs as a png in the working directory

def do_tplot():
    neuron = VDP()
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
