#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, xlim, ylim, savefig
from numpy import argsort, abs, mean
from numpy.fft import fft, fftfreq
from Python.NeuroFizzMath import solutionGenerator


# Most complicated example, time plot Hodgkin-Huxley


def do_tplot():
    solution = solutionGenerator('HH', 'rk4')
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    timeArray = solution[1]
    figure()
    plot(timeArray, -membranePotential)
    title('Hodgkin-Huxley')
    xlabel('Time')
    ylabel('Membrane Potential')
    savefig('HH_tplot.png')
    return


# Second example for phase plot of same model


def do_pplot():
    solution = solutionGenerator('HH', 'rk4')
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


def do_fftplot():
    solution = solutionGenerator('HH', 'rk4')
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    timeArray = solution[1]
    Y = mean(membranePotential)                  # determine DC component of signal
    X = membranePotential - Y       # subtract DC component from PS to get rid of peak at 0
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
    savefig('HH_fftplot.png')
    return


if __name__ == '__main__':
    print(do_tplot())
    print(do_pplot())
    print(do_fftplot())
