#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt

# Do some test plots and save outputs as a png in the working directory


class Plotter:
    def __init__(self):


    def PlotIt():
        for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
            model = i
            for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.eqns),
                  ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.eqns),
                  rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.eqns)]:
                solved = j
                return solved.tsp, solved.Nsize, solved.X, solved.X[0], solved.X[:, 0], solved.evaluate()



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
