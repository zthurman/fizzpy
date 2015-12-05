#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import matplotlib.pyplot as plt

# Do some test plots and save outputs as png files in the working directory


class DataMaker:
    def __init__(self, name='alldadatas'):
        self.name = name

    def datagenerator(self):
        for i in [VDP(), FN(), ML(), IZ(), HR(), HH()]:
            model = i
            for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
                  rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)]:
                solved = j
                return solved.tsp, solved.N_size, solved.X, solved.X[0], solved.X[:, 0], solved.evaluate()

data = DataMaker()
print data.datagenerator()

class Plotter(DataMaker):
    def __init__(self, name = 'alldaplots'):
        self.name = name

    # def do_tplot():
    #     neuron =
    #     solvedneuron = ord2(neuron.name, neuron.xaxis, neuron.yaxis, neuron.x0, neuron.dt, neuron.t_array, neuron.eqns)
    #     membranepotential = solvedneuron.evaluate()
    #     plt.figure()
    #     plt.plot(solvedneuron.tsp, -membranepotential[:, 0])
    #     plt.title(solvedneuron.modelname)
    #     plt.xlabel(solvedneuron.xaxis)
    #     plt.ylabel(solvedneuron.yaxis)
    #     plt.savefig('%s_tplot.png' % solvedneuron.model_name)
    #     plt.show()
    #     return

