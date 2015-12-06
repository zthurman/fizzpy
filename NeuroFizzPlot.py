#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
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


class DataMaker:
    def __init__(self, name='alldadatas', models=[VDP(), FN(), ML(), IZ(), HR(), HH()], solved=[]):
        self.name = name
        self.models = models
        self.solved = solved
#
#     def datagenerator(self):
#         for i in self.models:
#             model = i
#             return model
#             # for j in [euler(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
#             #       ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations),
#             #       rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)]:
#             #     self.solved = j
#             #     return self.solved.tsp, self.solved.N_size, self.solved.X, self.solved.X[0],
#             #     self.solved.X[:, 0], self.solved.evaluate()
#
# data = DataMaker()
# print data.datagenerator()
#
# class Plotter(DataMaker):
#     def __init__(self, name = 'alldaplots'):
#         self.name = name
#
#     # def do_tplot():
#     #     neuron =
#     #     solvedneuron = ord2(neuron.name, neuron.xaxis, neuron.yaxis, neuron.x0, neuron.dt, neuron.t_array, neuron.eqns)
#     #     membranepotential = solvedneuron.evaluate()
#     #     plt.figure()
#     #     plt.plot(solvedneuron.tsp, -membranepotential[:, 0])
#     #     plt.title(solvedneuron.modelname)
#     #     plt.xlabel(solvedneuron.xaxis)
#     #     plt.ylabel(solvedneuron.yaxis)
#     #     plt.savefig('%s_tplot.png' % solvedneuron.model_name)
#     #     plt.show()
#     #     return

