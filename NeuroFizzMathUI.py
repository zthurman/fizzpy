#!/usr/bin/env python
# NeuroFizzMath
# Copyright (C) 2015 Zechariah Thurman
# User interface for NeuroFizzMath program based on:

# embedding_in_qt4.py --- Simple Qt4 application embedding matplotlib canvases
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale

from __future__ import unicode_literals
from NeuroFizzMath import *
import numpy as np
import sys
import os
import random
from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

def rk4(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):
        tsp = np.arange(t0, t1, dt)
        Nsize = np.size(tsp)
        X = np.empty((Nsize, np.size(x0)))
        X[0] = x0
        for i in range(1, Nsize):
            k1 = ng(X[i-1],tsp[i-1])
            k2 = ng(X[i-1] + dt/2*k1, tsp[i-1] + dt/2)
            k3 = ng(X[i-1] + dt/2*k2, tsp[i-1] + dt/2)
            k4 = ng(X[i-1] + dt*k3, tsp[i-1] + dt)
            X[i] = X[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return X

"""def FN(x,t, a = 0.75, b = 0.8, c = 3,  i = -1.476):
    return np.array([c*(x[0]+ x[1]- x[0]**3/3 + i),
                    -1/c*(x[0]- a + b*x[1])])"""

class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    X = FN(Neuron)
    X.rk4
    def compute_initial_figure(self):
        X = FN(Neuron)
        X.rk4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        #X = rk4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    """def __init__(self, *args, **kwargs):
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
        self.draw()"""


class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")


        # file menu
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # model menu
        self.model_menu = QtGui.QMenu('&Models', self)
        self.model_menu.addAction('&Fitzhugh-Nagumo', self.fitzhughNagumo)
        self.model_menu.addAction('&Morris-Lecar', self.morrisLecar)
        self.model_menu.addAction('&Izikevich', self.izhikevich)
        self.model_menu.addAction('&Hindmarsh-Rose', self.hindmarshRose)
        self.model_menu.addAction('&Hodgkins-Huxley', self.hodgkinsHuxley)
        self.menuBar().addMenu(self.model_menu)

        # help menu
        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.help_menu.addAction('&About', self.about)
        self.help_menu.addAction('&Copyright', self.copyright)
        self.menuBar().addMenu(self.help_menu)

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=5, height=7, dpi=90)
        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=7, dpi=90)
        l.addWidget(sc)
        l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def buttonClicked(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def fitzhughNagumo(self):
        QtGui.QMessageBox.about(self, "Fitzhugh-Nagumo",
"""Fitzhugh-Nagumo

The Fitzhugh-Nagumo model is a system
of two coupled nonlinear differential
equations.
""")

    def morrisLecar(self):
        QtGui.QMessageBox.about(self, "Morris-Lecar",
"""Morris-Lecar

The Morris-Lecar model is a system
of two coupled nonlinear differential
equations.
""")

    def izhikevich(self):
        QtGui.QMessageBox.about(self, "Izhikevich",
"""Izhikevich

The Izhikevich model is a system
of two coupled nonlinear differential
equations.
""")

    def hindmarshRose(self):
        QtGui.QMessageBox.about(self, "Hindmarsh-Rose",
"""Hindmarsh-Rose

The Hindmarsh-Rose model is a system
of three coupled nonlinear differential
equations.
""")

    def hodgkinsHuxley(self):
        QtGui.QMessageBox.about(self, "Hodgkins-Huxley",
"""Hodgkins-Huxley

The Hodgkins-Huxley model is a system
of four coupled nonlinear differential
equations.
""")

    def about(self):
        QtGui.QMessageBox.about(self, "About",
"""NeuroFizzMath

This application allows the user to play with
different models of point neurons. Plots of
the membrane potential over time, phase plots
and FFTs are available.

Supported models are Fitzhugh-Nagumo, Morris-
Lecar, Izikevich, Hindmarsh-Rose and Hodgkins-
Huxley.
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

qApp = QtGui.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("NeuroFizzMath")
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()