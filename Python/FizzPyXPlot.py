#!/usr/bin/env python
# FizzPyX - FizzPyXPlot
# Copyright (C) 2017 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, xlim, ylim, savefig
from numpy import argsort, abs, mean, arange, argmax
from numpy.fft import fft, rfft, fftfreq
# from Python.FizzPyXFreq import InputtoFrequencyGen
from Python.FizzPyX import solutionGenerator, CoupledOscillatorsGen, FitzhughNagumoGen


# Time series plot

def do_tplot(modelname, solvername, plotname=None, xaxis=None, yaxis=None):
    solution = solutionGenerator(modelname, solvername)
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


# Input Stimulus to Frequency Plot

def InputtoFrequencyGen():
    freq = []
    inputs = []
    for I in arange(0.398, 0.539, 0.001):
        I *= -1
        solution = FitzhughNagumoGen('FN', 'ord2', i=I)
        solutionArray = solution[0]
        membranePotential = solutionArray[:, 0]
        Y = mean(membranePotential)  # determine DC component of signal
        X = membranePotential - Y  # subtract DC component from PS to get rid of peak at 0
        fdata = X.size
        ps = abs(rfft(X)) ** 2
        time_step = 1 / 30
        freqs = fftfreq(int(fdata / 2 - 1), time_step)

        locpeak = argmax(ps)  # Find its location
        maxfreq = freqs[locpeak]  # Get the actual frequency value

        freq.append(maxfreq)
        inputs.append(I)

    return inputs, freq


if __name__ == '__main__':
    # print(do_tplot('CO', 'ord2', plotname='Coupled Oscillators - Beats', xaxis='Time', yaxis='Mass Position'))
    # print(do_pplot('HH', 'rk4'))
    # print(do_psplot('HH', 'rk4'))

    data = InputtoFrequencyGen()
    plot(abs(data[0]), data[1])
    title('Cheese')
    xlabel('x')
    ylabel('y')
    # xlim(0, 1)
    # ylim(0, 2.5e9)
    savefig('cheese_tplot.png')
