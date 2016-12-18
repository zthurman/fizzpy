#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, xlim, ylim, savefig
from numpy import argsort, abs, mean, array
from numpy.fft import fft, fftfreq
from Python.NeuroFizzMath import solutionGenerator


# Most complicated example, time plot Hodgkin-Huxley


def do_tplot(modelname, solvername):
    solution = solutionGenerator(modelname, solvername, inits=array([0, 0, 0.5, 0]), endtime=200)
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    membranePotential1 = solutionArray[:, 2]
    timeArray = solution[1]
    figure()
    plot(timeArray, membranePotential, color='red')
    plot(timeArray, membranePotential1)
    title('Hodgkin-Huxley')
    xlabel('Time')
    ylabel('Membrane Potential')
    savefig('HH_tplot.png')
    return


# Second example for phase plot of same model


def do_pplot(modelname, solvername):
    solution = solutionGenerator(modelname, solvername)
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    KgatingVariable = solutionArray[:, 1]
    figure()
    plot(KgatingVariable, membranePotential)
    title('Hodgkin-Huxley')
    xlabel('Potassium Gating Variable')
    ylabel('Membrane Potential')
    savefig('HH_pplot.png')
    return


# Third example for fft plot of same model


def do_psplot(modelname, solvername):
    solution = solutionGenerator(modelname, solvername)
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    timeArray = solution[1]
    Y = mean(membranePotential)                 # determine DC component of signal
    X = membranePotential - Y                   # subtract DC component from PS to get rid of peak at 0
    fdata = X.size
    ps = abs(fft(X))**2
    time_step = 1 / 30
    freqs = fftfreq(int(fdata/2 - 1), time_step)
    idx = argsort(freqs)
    figure()
    plot(freqs[idx], ps[idx])
    title('Power Spectrum of Membrane Potential Signal')
    xlabel('Frequency')
    ylabel('Power')
    xlim(0, 1)
    ylim(0, 2.5e9)
    savefig('HH_psplot.png')
    return


if __name__ == '__main__':
    print(do_tplot('CO', 'ord2'))
    # print(do_pplot('HH', 'rk4'))
    # print(do_psplot('HH', 'rk4'))
