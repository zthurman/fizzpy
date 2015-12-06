#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, LIF, FN, ML, IZ, HR, HH
from NeuroFizzSolver import ord2, rk4
import matplotlib.pyplot as plt
import numpy as np

# Most basic example, generate Van der Pol oscillator time plot from data
# using second order solver


def do_tplot():
    model = VDP()
    solvedmodel = ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)
    dynamicalvariable = solvedmodel.evaluate()
    plt.figure()
    plt.plot(solvedmodel.tsp, dynamicalvariable[:, 0])
    plt.title(solvedmodel.model_name)
    plt.xlabel(solvedmodel.xaxis)
    plt.ylabel(solvedmodel.yaxis)
    plt.savefig('%s_tplot.png' % solvedmodel.model_name)
    plt.show()
    return

print do_tplot()

# Second basic example, generate Van der Pol oscillator phase plot from data
# using second order solver


def do_pplot():
    model = VDP()
    solvedmodel = ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)
    dynamicalvariable = solvedmodel.evaluate()
    plt.figure()
    plt.plot(dynamicalvariable[:, 1], dynamicalvariable[:, 0])
    plt.title(solvedmodel.model_name)
    plt.xlabel(solvedmodel.xaxis)
    plt.ylabel(solvedmodel.yaxis)
    plt.savefig('%s_pplot.png' % solvedmodel.model_name)
    plt.show()
    return

print do_pplot()

def do_fftplot():
    model = VDP()
    solvedmodel = ord2(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)
    X = solvedmodel.evaluate()
    Y = np.mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from PS to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[4:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[4:, 0])/2 - 1), time_step)
    idx = np.argsort(freqs)
    plt.figure()
    plt.plot(freqs[idx], ps[idx])
    plt.title(solvedmodel.model_name)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.xlim(0, 0.8)
    plt.ylim(0, 2.5e7)
    plt.savefig('%s_fftplot.png' % solvedmodel.model_name)
    plt.show()
    return

print do_fftplot()

# Now we do ALL OF THE THINGS! (With the rk4 model to make sure ML doesn't look funky)


def do_ALLPLOTS():
    for i in [VDP(), LIF(), FN(), ML(), IZ(), HR(), HH()]:
        model = i
        solvedmodel = rk4(model.name, model.xaxis, model.yaxis, model.x0, model.dt, model.t_array, model.equations)
        dynamicalvariable = solvedmodel.evaluate()
        plt.figure()
        if solvedmodel.model_name == 'Hodgkin-Huxley':
            plt.plot(solvedmodel.tsp, -dynamicalvariable[:, 0])   # Because this model is weird like that
        else:
            plt.plot(solvedmodel.tsp, dynamicalvariable[:, 0])
        plt.title(solvedmodel.model_name)
        plt.xlabel(solvedmodel.xaxis)
        plt.ylabel(solvedmodel.yaxis)
        plt.savefig('%s_tplot.png' % solvedmodel.model_name)
        plt.show()
    return

print do_ALLPLOTS()

