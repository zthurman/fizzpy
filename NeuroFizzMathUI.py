#!/usr/bin/env python
# NeuroFizzMath
# Neuroscience | Physics | Mathematics Toolkit

# Copyright (C) 2015 Zechariah Thurman

# User interface for NeuroFizzMath Toolkit with help from:
# http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html

from __future__ import unicode_literals
from NeuroFizzMath import *    #ord2, rk4, VDP, FN, ML, IZ, HR, HH, RD, L, R
import numpy as np
import sys
import os
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
import itertools

#   Choose PyQt4 or PySide, be aware of the licensing cost of building a PyQt application. Compare
# that to the lack of licensing fee for commercial applications with PySide.

use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

progname = os.path.basename(sys.argv[0])
progversion = "0.12"


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

    def compute_initial_figure(self, X):
        pass

# static canvas methods

class StaticVDPCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = VDP("van der Pol oscillator")
        X = ord2(x0 = np.array([0.01, 0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('X Dynamical Variable')
        self.axes.set_title('van der Pol oscillator')

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
        X = rk4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.1, ng = X.model)
        t = np.arange(0, 800, 0.1)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Hindmarsh-Rose')

class StaticHHCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = HH("Hodgkins-Huxley")
        X = rk4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = X.model)
        t = np.arange(0, 100, 0.02)
        self.axes.plot(t, -X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Membrane Potential')
        self.axes.set_title('Hodgkins-Huxley')

class StaticRDCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = RD("Rikitake Dynamo")
        X = rk4(x0 = np.array([-1.4, -1, -1, -1.4, 2.2, -1.5]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Geomagnetic Polarity')
        self.axes.set_title('Rikitake Dynamo')

class StaticLCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = L("Lorenz Equations")
        X = rk4(x0 = np.array([1.0, 2.0, 1.0]), t1 = 100,dt = 0.01, ng = X.model)
        t = np.arange(0, 100, 0.01)
        self.axes.plot(t, X[:,0])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('X Dynamical Variable')
        self.axes.set_title('Lorenz Equations')

class StaticRCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        X = R("Robbins Equations")
        X = rk4(x0 = np.array([0.00032,0.23,0.51]), t1 = 200,dt = 0.1, ng = X.model)
        t = np.arange(0, 200, 0.1)
        self.axes.plot(t, X[:,2])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Geomagnetic Polarity')
        self.axes.set_title('Robbins Equations')

class StaticMplCanvas(MyMplCanvas):
    def compute_initial_figure(self):
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

# nav toolbar

class MultiTabNavTool(NavigationToolbar):
    #====================================================================================================
    def __init__(self, canvases, tabs, parent=None):
        self.canvases = canvases
        self.tabs = tabs

        NavigationToolbar.__init__(self, canvases[0], parent)

    #====================================================================================================
    def get_canvas(self):
        return self.canvases[self.tabs.currentIndex()]

    def set_canvas(self, canvas):
        self._canvas = canvas

    canvas = property(get_canvas, set_canvas)

class MplMultiTab(QtGui.QMainWindow):
    #====================================================================================================
    def __init__(self, parent=None, figures=None, labels=None):
        qt.QMainWindow.__init__(self, parent)

        self.main_frame = QtGui.QWidget()
        self.tabWidget = QtGui.QTabWidget( self.main_frame )
        self.create_tabs( figures, labels )

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = MultiTabNavTool(self.canvases, self.tabWidget, self.main_frame)


        self.vbox = vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.tabWidget)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    #====================================================================================================
    def create_tabs(self, figures, labels ):

        if labels is None:      labels = []
        figures =  [Figure()] if figures is None else figures     #initialise with empty figure in first tab if no figures provided
        self.canvases = [self.add_tab(fig, lbl)
                            for (fig, lbl) in itertools.zip_longest(figures, labels) ]

    #====================================================================================================
    def add_tab(self, fig=None, name=None):
        '''dynamically add tabs with embedded matplotlib canvas with this function.'''

        # Create the mpl Figure and FigCanvas objects.
        if fig is None:
            fig = Figure()
            ax = fig.add_subplot(111)

        canvas = fig.canvas if fig.canvas else FigureCanvas(fig)
        canvas.setParent(self.tabWidget)
        canvas.setFocusPolicy( QtCore.Qt.ClickFocus )

        #self.tabs.append( tab )
        name = 'Tab %i'%(self.tabWidget.count()+1) if name is None else name
        self.tabWidget.addTab(canvas, name)

        return canvas

class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.layout = QtGui.QGridLayout()

        self.setGeometry(350, 350, 850, 550)

        # Sub-label bar

        self.menuBar().addSeparator()
        self.subLabel = QtGui.QMenu('Neuroscience | Physics | Mathematics Toolkit', self)
        self.statusBar().showMessage("Click some buttons!!!", 2000)
        self.menuBar().addMenu(self.subLabel)
        self.menuBar().addSeparator()

        # tool bar action list

        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'Exit', self)
        exitAction.triggered.connect(QtGui.qApp.quit)

        VDPAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'VDP', self)
        VDPAction.triggered.connect(self.draw_VDPcanvas)

        FNAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'FN', self)
        FNAction.triggered.connect(self.draw_FNcanvas)

        MLAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'ML', self)
        MLAction.triggered.connect(self.draw_MLcanvas)

        IZAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'IZ', self)
        IZAction.triggered.connect(self.draw_IZcanvas)

        HRAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HR', self)
        HRAction.triggered.connect(self.draw_HRcanvas)

        HHAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'HH', self)
        HHAction.triggered.connect(self.draw_HHcanvas)

        RDAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'RD', self)
        RDAction.triggered.connect(self.draw_RDcanvas)

        LAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'L', self)
        LAction.triggered.connect(self.draw_Lcanvas)

        RAction = QtGui.QAction(QtGui.QIcon.fromTheme('dude'), 'R', self)
        RAction.triggered.connect(self.draw_Rcanvas)

        aboutAction = QtGui.QAction(QtGui.QIcon.fromTheme('about'), 'About', self)
        aboutAction.triggered.connect(self.about)

        copyrightAction = QtGui.QAction(QtGui.QIcon.fromTheme('copyright'), 'Copyright', self)
        copyrightAction.triggered.connect(self.copyright)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)

        self.toolbar = self.addToolBar('van der Pol')
        self.toolbar.addAction(VDPAction)

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

        # final focus setting and other shiznats for main window

        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()

        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.statusBar().showMessage("The Diff EQ playground!", 2000)
        self.centralWidget.setFocus()

    def draw_centralWidget(self):
        def __init__(self):
            self.centralWidget.close()
            self.centralWidget = QtGui.QWidget(self)
            self.tabWidget = QtGui.QTabWidget(self.centralWidget)
            self.vbox = vbox = QtGui.QGridLayout()
            vbox.addWidget(self.tabWidget, 0, 0)
            vbox.addWidget(self.centralWidget, 2, 0)

    def draw_VDPcanvas(self, draw_centralWidget):
        l = QtGui.QGridLayout(self.centralWidget)
        sc = StaticVDPCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The van der Pol oscillator!", 2000)

    def draw_FNcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticFNCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Fitzhugh-Nagumo model!", 2000)

    def draw_MLcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticMLCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Morris-Lecar model!", 2000)

    def draw_IZcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticIZCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Izhikevich model!", 2000)

    def draw_HRcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticHRCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Hindmarsh-Rose model!", 2000)

    def draw_HHcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticHHCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Hodgkins-Huxley model!", 2000)

    def draw_RDcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticRDCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Rikitake Dynamo!", 2000)

    def draw_Lcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticLCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
        self.statusBar().showMessage("The Lorenz equations!", 2000)

    def draw_Rcanvas(self, draw_centralWidget):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        l = QtGui.QVBoxLayout(self.centralWidget)
        sc = StaticRCanvas(self.centralWidget, width=7, height=7, dpi=90)
        l.addWidget(sc)
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
    aw.setWindowTitle("NeuroFizzMath" + ' - ' + progversion)
    aw.show()
    sys.exit(qApp.exec_())
