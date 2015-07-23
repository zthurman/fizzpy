#!/usr/bin/env python
# NeuroFizzMath
# Neuroscience | Physics | Mathematics Toolkit

# Copyright (C) 2015 Zechariah Thurman

from __future__ import unicode_literals
from NeuroFizzMath import ord2, rk4, VDP, EPSP, FN, ML, IZ, HR, HH, RD, L, R
import numpy as np
import sys
import os
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
import itertools
from PyQt4 import QtGui, QtCore, QtWebKit

#   Choose PyQt4 or PySide, be aware of the licensing cost of building a PyQt application. Compare
# that to the lack of licensing fee for commercial applications with PySide.

"""use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore"""

progname = os.path.basename(sys.argv[0])
progversion = "0.15"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
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

    def compute_initial_figure(self, xlabel = '', ylabel = '', title = ''):
        X = self.system()
        x0 = X.x0
        X = rk4(x0, t1 = 100,dt = 0.02, ng = X.model)
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
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'X Dynamical Variable', title = 'van der Pol oscillator'):
        X = self.system()
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100, dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(X[:,1], X[:,0])
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

class StaticEPSPCanvas(MyMplCanvas):
    ylabel='Membrane Potential'
    def compute_initial_figure(self):
        X = EPSP("EPSP")
        X = X.model
        t = np.arange(0, 10, 0.01)
        self.plt.plot(t, X[0,:])
        #self.plt.plot(t, X[:,1]*5, 'r--')
        #self.plt.plot(t, X[:,2]/5, 'k:')
        self.axes.set_title('EPSP')

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

class StaticHRCanvas(MyMplCanvas):
    system = HR
    def compute_initial_figure(self, xlabel = 'Time', ylabel = 'Membrane Potential', title = 'Hindmarsh-Rose'):
        X = self.system()
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.1, ng = X.model)
        t = np.arange(0, 800, 0.1)
        self.axes.plot(t, X[:,0])
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

# main window

class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.layout = QtGui.QVBoxLayout()

        self.setGeometry(350, 350, 850, 550)

        # Sub-label bar

        self.menuBar().addSeparator()
        self.subLabel = QtGui.QMenu('Neuroscience | Physics | Mathematics Toolkit', self)
        self.statusBar().showMessage("Click some buttons!!!", 2000)
        self.menuBar().addMenu(self.subLabel)
        self.menuBar().setToolTip('Troll button!')
        self.menuBar().addSeparator()

        # tool bar action list

        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'Exit', self)
        exitAction.triggered.connect(QtGui.qApp.quit)
        exitAction.setToolTip('Exit the program')

        VDPAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'VDP', self)
        VDPAction.triggered.connect(self.draw_VDPcanvas)
        VDPAction.setToolTip('van der Pol oscillator')

        EPSPAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'EPSP', self)
        EPSPAction.triggered.connect(self.draw_EPSPcanvas)
        EPSPAction.setToolTip('Excitatory Post-synaptic potential')

        FNAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'FN', self)
        FNAction.triggered.connect(self.draw_FNcanvas)
        FNAction.setToolTip('Fitzhugh-Nagumo model')

        MLAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'ML', self)
        MLAction.triggered.connect(self.draw_MLcanvas)
        MLAction.setToolTip('Morris-Lecar model')

        IZAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'IZ', self)
        IZAction.triggered.connect(self.draw_IZcanvas)
        IZAction.setToolTip('Izhikevich model')

        HRAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HR', self)
        HRAction.triggered.connect(self.draw_HRcanvas)
        HRAction.setToolTip('Hindmarsh-Rose model')

        HHAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HH', self)
        HHAction.triggered.connect(self.draw_HHcanvas)
        HHAction.setToolTip('Hodgkins-Huxley model')

        RDAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'RD', self)
        RDAction.triggered.connect(self.draw_RDcanvas)
        RDAction.setToolTip('Rikitake dynamo')

        LAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'L', self)
        LAction.triggered.connect(self.draw_Lcanvas)
        LAction.setToolTip('Lorenz equations')

        RAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'R', self)
        RAction.triggered.connect(self.draw_Rcanvas)
        RAction.setToolTip('Robbins model')

        aboutAction = QtGui.QAction(QtGui.QIcon.fromTheme('about'), 'About', self)
        aboutAction.triggered.connect(self.about)

        copyrightAction = QtGui.QAction(QtGui.QIcon.fromTheme('copyright'), 'Copyright', self)
        copyrightAction.triggered.connect(self.copyright)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)

        self.toolbar = self.addToolBar('van der Pol')
        self.toolbar.addAction(VDPAction)

        self.toolbar = self.addToolBar('EPSP')
        self.toolbar.addAction(EPSPAction)

        self.toolbar = self.addToolBar('Fitzhugh-Nagumo')
        self.toolbar.addAction(FNAction)

        self.toolbar = self.addToolBar('Morris-Lecar')
        self.toolbar.addAction(MLAction)

        self.toolbar = self.addToolBar('Izhikevich')
        self.toolbar.addAction(IZAction)

        self.toolbar = self.addToolBar('Hindmarsh-Rose')
        self.toolbar.addAction(HRAction)

        self.toolbar = self.addToolBar('Hodgkins-Huxley')
        self.toolbar.addAction(HHAction)

        self.toolbar = self.addToolBar('Rikitake Dynamo')
        self.toolbar.addAction(RDAction)

        self.toolbar = self.addToolBar('Lorenz Equations')
        self.toolbar.addAction(LAction)

        self.toolbar = self.addToolBar('Robbins Equations')
        self.toolbar.addAction(RAction)

        self.toolbar = self.addToolBar('About')
        self.toolbar.addAction(aboutAction)

        self.toolbar = self.addToolBar('Copyright')
        self.toolbar.addAction(copyrightAction)

        # final focus setting and other shiznats for the rest of main window

        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()

        self.centralWidget = QtGui.QWidget(self)

        self.statusBar().showMessage("The Diff EQ playground!", 2000)

    def vdptpbutton_refresh(self, sc):
        self.centralWidget.close()
        sc = StaticVDPCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(sc)
        self.centralWidget.close()


    def draw_VDPcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the x dynamical variable over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the oscillator')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox1 = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox1)

        sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox1.addWidget(self.tpbutton)

        self.tpbutton.clicked.connect(self.vdptpbutton_refresh)
        self.layout.addWidget(sc)

        #sc1 = StaticPplotVDPCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox1.addWidget(self.ppbutton)
        #self.ppbutton.addAction(sc1)
        #self.ppbutton.clicked.connect(self.ppbutton)

        self.hbox1.addWidget(self.fftbutton)
        #self.layout1.addWidget(sc)

        self.layout2 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("http://goo.gl/0KXNw"))

        self.layout2.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The van der Pol oscillator!", 2000)

    def draw_EPSPcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)
        self.frame = QtGui.QFrame(self.tabs)
        layout = QtGui.QVBoxLayout(self.frame)

        sc = StaticEPSPCanvas(self.tab1, width=7, height=7, dpi=70)
        layout.addWidget(sc)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)

        self.tabs.setFixedWidth(850)
        self.tabs.setFixedHeight(450)

        self.centralWidget.setFocus()
        self.statusBar().showMessage("An Excitatory Post-synaptic Potential!", 2000)

    def draw_FNcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticFNCanvas(self.tab1, width=7, height=7, dpi=70)

        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout2 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("http://goo.gl/X9ISh"))

        self.layout2.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Fitzhugh-Nagumo model!", 2000)

    def draw_MLcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticMLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("http://goo.gl/F2dDcl"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Morris-Lecar model!", 2000)

    def draw_IZcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticIZCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("http://goo.gl/FcWxh"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Izhikevich model!", 2000)

    def draw_HRcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticHRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("https://goo.gl/M0x7lH"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Hindmarsh-Rose model!", 2000)

    def draw_HHcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticHHCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("https://goo.gl/mE88Oi"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Hodgkins-Huxley model!", 2000)

    def draw_RDcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticRDCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("https://goo.gl/sYOlZ"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Rikitake Dynamo!", 2000)

    def draw_Lcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("https://goo.gl/V3Yb77"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Lorenz equations!", 2000)

    def draw_Rcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('FFT Plot', self.tabs)
        self.fftbutton.setToolTip('Generate a fast Fourier transform for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        sc = StaticRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)
        self.layout.addWidget(sc)

        self.layout3 = QtGui.QVBoxLayout(self.tab3)

        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("https://goo.gl/L99Uha"))

        self.layout3.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.statusBar().showMessage("The Robbins Dynamo!", 2000)

    def buttonClicked(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtGui.QMessageBox.about(self, "About",
        """NeuroFizzMath

        This application allows the user to play with
        different ODEs. Plots of the models allow the
        user to get a feel for how the different sys-
        tems behave.

        Supported models are Fitzhugh-Nagumo, Morris-
        Lecar, Izikevich, Hindmarsh-Rose and Hodgkins-
        Huxley, the Rikitake Dynamo, the Lorenz Equa-
        tions and the Robbins Model.
        """)

    def copyright(self):
        QtGui.QMessageBox.about(self, "Copyright",
        """Copyright (C) 2015 by Zechariah Thurman

        Permission is hereby granted, free of charge,
        to any person obtaining a copy of this software
        and associated documentation files (the
        "Software"), to deal in the Software without
        restriction, including without limitation the
        rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of
        the Software, and to permit persons to whom the
        Software is furnished to do so, subject to the
        following conditions:

        The above copyright notice and this permission
        notice shall be included in all copies or
        substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
        WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
        INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
        AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
        OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
        DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
        OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
        OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
        OR OTHER DEALINGS IN THE SOFTWARE.
        """
        )


if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('NeuroFizzMath' + ' - ' + progversion )
    aw.show()
    sys.exit(qApp.exec_())
