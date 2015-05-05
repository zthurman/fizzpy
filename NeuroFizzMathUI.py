#!/usr/bin/env python
# NeuroFizzMath
# Copyright (C) 2015 Zechariah Thurman
# User interface for NeuroFizzMath with help from:

# embedding_in_qt4.py --- Simple Qt4 application embedding matplotlib canvases
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale

from __future__ import unicode_literals
from NeuroFizzMath import rk4, FN, ML, IZ, HR, HH, RD, W, L
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
        # self.fig.clear()
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

    def compute_initial_figure(self, X):
        pass

# static canvas methods

class StaticFNCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = FN("Fitzhugh-Nagumo")
        X = rk4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Fitzhugh-Nagumo')

class StaticMLCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = ML("Morris-Lecar")
        X = rk4(x0 = np.array([0,0]), t1 = 1000,dt = 0.30, ng = X.model)
        t = np.arange(0, 1000, 0.30)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Morris-Lecar')

class StaticIZCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = IZ("Izhikevich")
        X = rk4(x0 = np.array([0,0]), t1 = 300,dt = 0.1, ng = X.model)
        t = np.arange(0, 300, 0.1)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Izhikevich')

class StaticHRCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = HR("Hindmarsh-Rose")
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Hindmarsh-Rose')

class StaticRDCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = RD("Rikitake Dynamo")
        X = rk4(x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Geomagnetic Polarity')
        self.axes.set_title('Rikitake Dynamo')

class StaticMplCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        # soon to be generalized method that will take any model as arguments
        # provided by QT event actions
        #X = FN("Fitzhugh-Nagumo")
        X = L("Lorenz Eqns")
        X = rk4(x0 = np.array([1.0, 2.0, 1.0]) , t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('X Dynamical Variable')
        self.axes.set_title('Lorenz Equations')

# dynamic canvas method

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
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        # landing screen text
#        self.text = \
#                'Welcome to NeuroFizzMath! This is a project that is designed to \n\
#assist you with understanding numerical solutions to systems  \n\
#    of differential equations.'
        self.setGeometry(300, 300, 800, 500)

        # file menu
        self.file_menu = QtGui.QMenu('File', self)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # model menu
        self.model_menu = QtGui.QMenu('Models', self)
        self.menuBar().addSeparator()
        self.model_menu.addAction('Fitzhugh-Nagumo', self.fitzhughNagumo)
        self.model_menu.addAction('Morris-Lecar', self.morrisLecar)
        self.model_menu.addAction('Izikevich', self.izhikevich)
        self.model_menu.addAction('Hindmarsh-Rose', self.hindmarshRose)
        self.model_menu.addAction('Hodgkins-Huxley', self.hodgkinsHuxley)
        self.model_menu.addAction('Rikitake Dynamo', self.rikitakeDynamo)
        self.model_menu.addAction('Lorenz Equations', self.lorenzEqns)
        self.model_menu.addAction('Robbins Model', self.robbins)
        self.menuBar().addMenu(self.model_menu)

        # help menu
        self.help_menu = QtGui.QMenu('Help', self)
        self.menuBar().addSeparator()
        self.help_menu.addAction('&About', self.about, QtCore.Qt.CTRL + QtCore.Qt.Key_A)
        self.help_menu.addAction('&Copyright', self.copyright, QtCore.Qt.CTRL + QtCore.Qt.Key_C)
        self.menuBar().addMenu(self.help_menu)

        # tool bar action list

        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'Exit', self)
        exitAction.triggered.connect(QtGui.qApp.quit)

        FNAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'FN', self)
        FNAction.connect(FNAction,QtCore.SIGNAL('triggered()'), self.draw_FNcanvas)

        MLAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'ML', self)
        MLAction.connect(MLAction,QtCore.SIGNAL('triggered()'), self.draw_MLcanvas)

        IZAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'IZ', self)
        IZAction.connect(IZAction,QtCore.SIGNAL('triggered()'), self.draw_IZcanvas)

        HRAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HR', self)
        HRAction.connect(HRAction,QtCore.SIGNAL('triggered()'), self.draw_HRcanvas)

        HHAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HH', self)
        HHAction.connect(HHAction,QtCore.SIGNAL('triggered()'), self.hodgkinsHuxley)

        RDAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'RD', self)
        RDAction.connect(RDAction,QtCore.SIGNAL('triggered()'), self.draw_RDcanvas)

        LAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'L', self)
        LAction.connect(LAction,QtCore.SIGNAL('triggered()'), self.lorenzEqns)

        RAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'R', self)
        RAction.connect(RAction,QtCore.SIGNAL('triggered()'), self.robbins)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)
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


        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()

        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.statusBar().showMessage("The Diff EQ playground!", 2000)

        # INTERFACE PORTION OF CANVAS DISPLAYED - START

        #l = QtGui.QVBoxLayout(self.main_widget)
        #sc = StaticMplCanvas(self.main_widget, width=7, height=7, dpi=90)
        #dc = DynamicMplCanvas(self.main_widget, width=7, height=7, dpi=90)
        #l.addWidget(sc)
        #l.addWidget(dc)

        # *NOTE* We want a Qt event driven conditional here to select
        # between the different models to display on the static canvas
        # this will allow for the UI to determine which plot shows

        # Probably a good idea to do this with toolbar for starters

        # INTERFACE PORTION OF CANVAS DISPLAYED - END

    def draw_FNcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticFNCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)

    def draw_MLcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticMLCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)

    def draw_IZcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticIZCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)

    def draw_HRcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticHRCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)

    def draw_RDcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticRDCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)

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

        For more details check out:
        http://goo.gl/qMu6eb
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
        equations and four functions.
        """)

    def rikitakeDynamo(self):
        QtGui.QMessageBox.about(self, "Hodgkins-Huxley",
        """Rikitake Dynamo

        The Rikitake Dynamo is a system
        of six coupled nonlinear differential
        equations that govern the phenom-
        enon of geomagnetic polarity reversal.
        """)

    def lorenzEqns(self):
        QtGui.QMessageBox.about(self, "Lorenz Equations",
        """Lorenz Equations

        The Lorenz Equations are a system
        of three coupled nonlinear differential
        equations that govern atmospheric
        convection behavior.
        """)

    def robbins(self):
        QtGui.QMessageBox.about(self, "Robbins Equations",
        """Robbins Equations

        The Robbins Equations are a system
        of three coupled nonlinear differential
        equations that model geomagnetic
        polarity reversal.
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

    # landing screen text events

    """def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawText(event, qp)
        qp.end()

    def drawText(self, event, qp):

        qp.setPen(QtGui.QColor(168, 34, 3))
        qp.setFont(QtGui.QFont('Decorative', 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignCenter, self.text)"""

if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.setWindowTitle("NeuroFizzMath" + ' ' + progversion)
    aw.show()
    sys.exit(qApp.exec_())
    #qApp.exec_()
