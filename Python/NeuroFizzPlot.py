#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, xlim, ylim, savefig
from numpy import argsort, abs, mean, array
from numpy.fft import fft, fftfreq
from Python.NeuroFizzMath import solutionGenerator, dimensionIdentifier


# Time series plot

def do_tplot(modelname, solvername, plotname=None, xaxis=None, yaxis=None):
    solution = solutionGenerator(modelname, solvername, inits=array([0, 0, 0.5, 0]), endtime=200)
    solutionArray = solution[0]
    firstarray = solutionArray[:, 0]
    secondarray = solutionArray[:, 2]
    timeArray = solution[1]
    figure()
    plot(timeArray, firstarray)
    plot(timeArray, secondarray)
    title(plotname)
    xlabel(xaxis)
    ylabel(yaxis)
    savefig('%s_tplot.png' % plotname)
    return


# Phase plot

def do_pplot(modelname, solvername, plotname=None, xaxis=None, yaxis=None):
    solution = solutionGenerator(modelname, solvername)
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    KgatingVariable = solutionArray[:, 1]
    figure()
    plot(KgatingVariable, membranePotential)
    title(plotname)
    xlabel(xaxis)
    ylabel(yaxis)
    savefig('%s_pplot.png' % plotname)
    return


# Power Spectrum

def do_psplot(modelname, solvername, plotname=None, xaxis=None, yaxis=None):
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
    title(plotname)
    xlabel(xaxis)
    ylabel(yaxis)
    xlim(0, 1)
    ylim(0, 2.5e9)
    savefig('%s_psplot.png' % plotname)
    return


if __name__ == '__main__':
    print(do_tplot('CO', 'euler', plotname='Coupled Oscillators - Beats', xaxis='Time', yaxis='Mass Position'))
    # print(do_pplot('HH', 'rk4'))
    # print(do_psplot('HH', 'rk4'))
