#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzCanvas
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import unicode_literals, division
from NeuroFizzSolver import euler, ord2, rk4, VDP
import numpy as np
import sys
import os
import random
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
import itertools
from PyQt4 import QtGui, QtCore, QtWebKit

class MyMplCanvas(FigureCanvas, solver = None):
    # Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.hold(False)
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self, solver, xlabel = '', ylabel = '', title = ''):
        X = self.system()
        x0 = X.inits()
        X = solver(x0, t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

# static canvas methods

class StaticNullCanvas(MyMplCanvas):
    system = VDP
    def compute_initial_figure(self, xlabel = '', ylabel = '', title = ''):
        X = self.system()
        X = np.arange(0, 100, 0.02)
        t = np.arange(0, 100, 0.02)
        self.axes.plot()
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticVDPCanvas(MyMplCanvas):
    system = VDP
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'X Dynamical Variable', title = 'van der Pol oscillator'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100, dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotVDPCanvas(MyMplCanvas):
    system = VDP
    def compute_initial_figure(self, xlabel = 'Y Dynamical Variable', ylabel = 'X Dynamical Variable', title = 'van der Pol oscillator'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100, dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotVDPCanvas(MyMplCanvas):
    system = VDP
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'van der Pol oscillator'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100, dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,1)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFNCanvas(MyMplCanvas):
    system = FN
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Fitzhugh-Nagumo'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotFNCanvas(MyMplCanvas):
    system = FN
    def compute_initial_figure(self, xlabel = 'Membrane Recovery Variable', ylabel = 'Membrane Potential', title = 'Fitzhugh-Nagumo'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotFNCanvas(MyMplCanvas):
    system = FN
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Fitzhugh-Nagumo'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100, dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,1)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticMLCanvas(MyMplCanvas):
    system = ML
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Morris-Lecar'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 1000,dt = 0.30, ng = X.model)
        t = np.arange(0, 1000, 0.30)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotMLCanvas(MyMplCanvas):
    system = ML
    def compute_initial_figure(self, xlabel = 'Membrane Recovery Variable', ylabel = 'Membrane Potential', title = 'Morris-Lecar'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 1000,dt = 0.30, ng = X.model)
        t = np.arange(0, 1000, 0.30)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotMLCanvas(MyMplCanvas):
    system = ML
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Morris-Lecar'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 1000, dt = 0.3, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,1)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticIZCanvas(MyMplCanvas):
    system = IZ
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Izhikevich'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 300,dt = 0.1, ng = X.model)
        t = np.arange(0, 300, 0.1)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotIZCanvas(MyMplCanvas):
    system = IZ
    def compute_initial_figure(self, xlabel = 'Recovery Variable', ylabel = 'Membrane Potential', title = 'Izhikevich'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 300,dt = 0.1, ng = X.model)
        t = np.arange(0, 300, 0.1)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotIZCanvas(MyMplCanvas):
    system = IZ
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Izhikevich'):
        X = self.system()
        X = rk4(x0 = np.array([0,0]), t1 = 1000, dt = 0.3, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,0.75)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticHRCanvas(MyMplCanvas):
    system = HR
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Hindmarsh-Rose'):
        X = self.system()
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.1, ng = X.model)
        t = np.arange(0, 800, 0.1)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlim(100,800)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotHRCanvas(MyMplCanvas):
    system = HR
    def compute_initial_figure(self, xlabel = 'Recovery Variable', ylabel = 'Membrane Potential', title = 'Hindmarsh-Rose'):
        X = self.system()
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.1, ng = X.model)
        t = np.arange(0, 800, 0.1)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotHRCanvas(MyMplCanvas):
    system = HR
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Hindmarsh-Rose'):
        X = self.system()
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.1, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,1)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticHHCanvas(MyMplCanvas):
    system = HH
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Hodgkins-Huxley'):
        X = self.system()
        X = ord2(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, -X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotHHCanvas(MyMplCanvas):
    system = HH
    def compute_initial_figure(self, xlabel = 'Recovery Variable', ylabel = 'Membrane Potential', title = 'Hodgkins-Huxley'):
        X = self.system()
        X = ord2(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotHHCanvas(MyMplCanvas):
    system = HH
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Hodgkins-Huxley'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,1)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticRDCanvas(MyMplCanvas):
    system = RD
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Geomagnetic Polarity', title = 'Rikitake Dynamo'):
        X = self.system()
        X = rk4(x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotRDCanvas(MyMplCanvas):
    system = RD
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Geomagnetic Polarity', title = 'Rikitake Dynamo'):
        X = self.system()
        X = rk4(x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(X[:,3], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotRDCanvas(MyMplCanvas):
    system = RD
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Rikitake Dynamo'):
        X = self.system()
        X = rk4(x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1 = 100,dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,2)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticLCanvas(MyMplCanvas):
    system = L
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'X Dynamical Variable', title = 'Lorenz Equations'):
        X = self.system()
        X = rk4(x0 = np.array([1.0, 2.0, 1.0]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotLCanvas(MyMplCanvas):
    system = L
    def compute_initial_figure(self, xlabel = 'X Dynamical Variable', ylabel = 'Z Dynamical Variable', title = 'Lorenz Equations'):
        X = self.system()
        X = rk4(x0 = np.array([1.0, 2.0, 1.0]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(X[:,0], X[:,2])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotLCanvas(MyMplCanvas):
    system = L
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Lorenz Equations'):
        X = self.system()
        X = rk4(x0 = np.array([1.0, 2.0, 1.0]), t1 = 100,dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,2)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticRCanvas(MyMplCanvas):
    system = R
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Geomagnetic Polarity', title = 'Robbins Equations'):
        X = self.system()
        X = rk4(x0 = np.array([0.00032,0.23,0.51]), t1 = 200,dt = 0.1, ng = X.model)
        t = np.arange(0, 200, 0.1)
        self.axes.plot(t, X[:,2])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticPplotRCanvas(MyMplCanvas):
    system = R
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Geomagnetic Polarity', title = 'Robbins Equations'):
        X = self.system()
        X = rk4(x0 = np.array([0.00032,0.23,0.51]), t1 = 200,dt = 0.1, ng = X.model)
        t = np.arange(0, 200, 0.1)
        self.axes.plot(X[:,0], X[:,2])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticFFTplotRCanvas(MyMplCanvas):
    system = R
    def compute_initial_figure(self, xlabel = 'Frequency', ylabel = 'Power', title = 'Robbins Equations'):
        X = self.system()
        X = rk4(x0 = np.array([0.00032,0.23,0.51]), t1 = 100,dt = 0.02, ng = X.model)
        Y = np.mean(X)    # determine DC component of signal
        X = X - Y      # subtract DC component from signal to get rid of peak at 0
        ps = np.abs(np.fft.fft(X[:,0]))**2
        time_step = 1 / 30
        freqs = np.fft.fftfreq(int((len(X[:, 0])/2 - 1)), time_step)
        idx = np.argsort(freqs)
        self.axes.plot(freqs[idx], ps[idx])
        self.axes.set_xlim(0,2)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

# dynamic canvas methods

class DynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()