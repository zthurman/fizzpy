#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzPlot
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv2


from __future__ import division
from NeuroFizzMath import solutionGenerator
import matplotlib.pyplot as plt
import numpy as np


# Most complicated example, time plot Hodgkin-Huxley


def do_tplot():
    solution = solutionGenerator('HH', 'rk4')
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    timeArray = solution[1]
    plt.figure()
    plt.plot(timeArray, -membranePotential)
    plt.title('Hodgkin-Huxley')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.savefig('HH_tplot.png')
    return


# Second example for phase plot of same model


def do_pplot():
    solution = solutionGenerator('HH', 'rk4')
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    KgatingVariable = solutionArray[:, 1]
    plt.figure()
    plt.plot(KgatingVariable, membranePotential)
    plt.title('Hodgkin-Huxley')
    plt.xlabel('Potassium Gating Variable')
    plt.ylabel('Membrane Potential')
    plt.savefig('HH_pplot.png')
    return


# Third example for fft plot of same model


def do_fftplot():
    solution = solutionGenerator('HH', 'rk4')
    solutionArray = solution[0]
    membranePotential = solutionArray[:, 0]
    timeArray = solution[1]
    Y = np.mean(membranePotential)                  # determine DC component of signal
    X = membranePotential - Y       # subtract DC component from PS to get rid of peak at 0
    fdata = X.size
    ps = np.abs(np.fft.fft(X))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)
    plt.figure()
    plt.plot(freqs[idx], ps[idx])
    plt.title('Power Spectrum of Membrane Potential Signal')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.xlim(0, 1)
    plt.ylim(0, 2.5e9)
    plt.savefig('HH_fftplot.png')
    return


if __name__ == '__main__':
    print(do_tplot())
    print(do_pplot())
    print(do_fftplot())
