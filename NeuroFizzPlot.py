#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, LIF, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt

# Most basic example, generate Van der Pol oscillator plot from data
# using second order solver


def do_tplot():
    neuron = VDP()
    solvedneuron = ord2(neuron.name, neuron.xaxis, neuron.yaxis, neuron.x0, neuron.dt, neuron.t_array, neuron.equations)
    membranepotential = solvedneuron.evaluate()
    plt.figure()
    plt.plot(solvedneuron.tsp, membranepotential[:, 0])
    plt.title(solvedneuron.model_name)
    plt.xlabel(solvedneuron.xaxis)
    plt.ylabel(solvedneuron.yaxis)
    plt.savefig('%s_tplot.png' % solvedneuron.model_name)
    plt.show()
    return

print do_tplot()

# Now we do ALL OF THE THINGS!


for i in [VDP(), LIF()]
