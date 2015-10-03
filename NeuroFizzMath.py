#!/usr/bin/env python
# NeuroFizzMath
# Neuroscience | Physics | Mathematics Toolkit

# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

# This is an implementation of matplotlib's pyqt5 backend

from __future__ import unicode_literals, division
from NeuroFizzModel import euler, ord2, rk4, VDP
import numpy as np
import sys
import os
import random
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
import itertools
from PyQt4 import QtGui, QtCore, QtWebKit

progname = os.path.basename(sys.argv[0])
progversion = "0.16"

# main window

class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

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

    # centralWidget drawcanvas methods, called by toolbar actions

    def vdptpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticVDPCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def vdpppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotVDPCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def vdpfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotVDPCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def vdpparam3_update(self):
        @VDP
        def model(self, x, t, mu = self.param3Edit):
            return np.array([x[1]/mu,
                             (-x[0] + x[1]*(1-x[0]**2))*mu])

    def Back(self):
        self.webview.back()

    def draw_VDPcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the x dynamical variable over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the oscillator')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.vdptpbutton_refresh)
        self.ppbutton.clicked.connect(self.vdpppbutton_refresh)
        self.fftbutton.clicked.connect(self.vdpfftbutton_refresh)

        self.layout1 = QtGui.QGridLayout(self.tab2)
        self.setLayout(self.layout1)
        self.layout1.spacing()

        self.font = QtGui.QFont()
        self.font.setBold(True)
        self.title = QtGui.QLabel('Modify the parameters below and go back to plotting')
        self.title.setFont(self.font)
        self.param = QtGui.QLabel('Periodic Forcing')
        self.paramEdit = QtGui.QLineEdit(self.tab2)
        self.paramEdit.setPlaceholderText('Periodic forcing value, defaults to 0')
        self.parambutton = QtGui.QPushButton('Update', self.tab2)
        self.parambutton.setToolTip('Update Periodic Forcing Value')
        self.param1 = QtGui.QLabel('Circularly Coupled Oscillators')
        self.param1Edit = QtGui.QLineEdit(self.tab2)
        self.param1Edit.setPlaceholderText('Number of circularly coupled oscillators, defaults to 0')
        self.param1button = QtGui.QPushButton('Update', self.tab2)
        self.param1button.setToolTip('Update Coupled Oscillators Count')
        self.param2 = QtGui.QLabel('Linearly Coupled Oscillators')
        self.param2Edit = QtGui.QLineEdit(self.tab2)
        self.param2Edit.setPlaceholderText('Number of linearly coupled oscillators, defaults to 0')
        self.param2button = QtGui.QPushButton('Update', self.tab2)
        self.param2button.setToolTip('Update Linearly Oscillators Count')
        self.param3 = QtGui.QLabel('Nonlinear Damping')
        self.param3Edit = QtGui.QLineEdit(self.tab2)
        self.param3Edit.setPlaceholderText('Nonlinear damping coefficient, defaults to 1')
        self.param3button = QtGui.QPushButton('Update', self.tab2)
        self.param3button.setToolTip('Update Nonlinear Damping Coefficient')
        self.layout1.addWidget(self.title, 1, 1, 1, 1)
        self.layout1.addWidget(self.param, 2, 1, 2, 1)
        self.layout1.addWidget(self.paramEdit, 2, 2, 2, 1)
        self.layout1.addWidget(self.parambutton, 2, 3, 2, 1)
        self.layout1.addWidget(self.param1, 3, 1, 2, 1)
        self.layout1.addWidget(self.param1Edit, 3, 2, 2, 1)
        self.layout1.addWidget(self.param1button, 3, 3, 2, 1)
        self.layout1.addWidget(self.param2, 4, 1, 2, 1)
        self.layout1.addWidget(self.param2Edit, 4, 2, 2, 1)
        self.layout1.addWidget(self.param2button, 4, 3, 2, 1)
        self.layout1.addWidget(self.param3, 5, 1, 2, 1)
        self.layout1.addWidget(self.param3Edit, 5, 2, 2, 1)
        self.layout1.addWidget(self.param3button, 5, 3, 2, 1)

        self.param3button.clicked.connect(self.vdpparam3_update)

        self.layout2 = QtGui.QVBoxLayout(self.tab3)
        self.hbox1 = QtGui.QHBoxLayout(self.tab3)
        self.layout2.addLayout(self.hbox1)
        self.webview = QtWebKit.QWebView(self.tab3)
        self.webview.load(QtCore.QUrl("http://goo.gl/0KXNw"))
        self.back = QtGui.QPushButton(self)
        self.back.setMinimumSize(35,30)
        self.back.setStyleSheet("font-size:23px;")
        self.back.clicked.connect(self.Back)
        self.back.setIcon(QtGui.QIcon().fromTheme("go-previous"))

        self.hbox1.addWidget(self.back)
        self.layout2.addWidget(self.webview)

        self.tabs.addTab(self.tab1, "Plots")
        self.tabs.addTab(self.tab2, "Model Parameters")
        self.tabs.addTab(self.tab3, "Background")

        self.setCentralWidget(self.tabs)
        self.centralWidget.setFocus()
        self.centralWidget.close()
        self.statusBar().showMessage("The van der Pol oscillator!", 2000)

    def fntpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFNCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def fnppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotFNCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def fnfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotFNCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_FNcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.fntpbutton_refresh)
        self.ppbutton.clicked.connect(self.fnppbutton_refresh)
        self.fftbutton.clicked.connect(self.fnfftbutton_refresh)

        self.layout1 = QtGui.QGridLayout(self.tab2)
        self.setLayout(self.layout1)
        self.font = QtGui.QFont()
        self.font.setBold(True)
        self.title = QtGui.QLabel('Modify the parameters below and go back to plotting')
        self.title.setFont(self.font)
        self.param = QtGui.QLabel('A')
        self.paramEdit = QtGui.QLineEdit(self.tab2)
        self.paramEdit.setPlaceholderText('A coefficient, defaults to 0.75')
        self.param1 = QtGui.QLabel('B')
        self.param1Edit = QtGui.QLineEdit(self.tab2)
        self.param1Edit.setPlaceholderText('B coefficient, defaults to 0.8')
        self.param2 = QtGui.QLabel('C')
        self.param2Edit = QtGui.QLineEdit(self.tab2)
        self.param2Edit.setPlaceholderText('C coefficient, defaults to 3')
        self.layout1.addWidget(self.title, 1, 1, 1, 1)
        self.layout1.addWidget(self.param, 2, 1, 2, 1)
        self.layout1.addWidget(self.paramEdit, 2, 2, 2, 1)
        self.layout1.addWidget(self.param1, 3, 1, 2, 1)
        self.layout1.addWidget(self.param1Edit, 3, 2, 2, 1)
        self.layout1.addWidget(self.param2, 4, 1, 2, 1)
        self.layout1.addWidget(self.param2Edit, 4, 2, 2, 1)


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

    def mltpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticMLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def mlppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotMLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def mlfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotMLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_MLcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.mltpbutton_refresh)
        self.ppbutton.clicked.connect(self.mlppbutton_refresh)
        self.fftbutton.clicked.connect(self.mlfftbutton_refresh)

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

    def iztpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticIZCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def izppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotIZCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def izfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotIZCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_IZcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.iztpbutton_refresh)
        self.ppbutton.clicked.connect(self.izppbutton_refresh)
        self.fftbutton.clicked.connect(self.izfftbutton_refresh)

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

    def hrtpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticHRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def hrppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotHRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def hrfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotHRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_HRcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.hrtpbutton_refresh)
        self.ppbutton.clicked.connect(self.hrppbutton_refresh)
        self.fftbutton.clicked.connect(self.hrfftbutton_refresh)

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

    def hhtpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticHHCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def hhppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotHHCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def hhfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotHHCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_HHcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.hhtpbutton_refresh)
        self.ppbutton.clicked.connect(self.hhppbutton_refresh)
        self.fftbutton.clicked.connect(self.hhfftbutton_refresh)

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

    def rdtpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticRDCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def rdppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotRDCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def rdfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotRDCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_RDcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.rdtpbutton_refresh)
        self.ppbutton.clicked.connect(self.rdppbutton_refresh)
        self.fftbutton.clicked.connect(self.rdfftbutton_refresh)

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

    def ltpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def lppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def lfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotLCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_Lcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.ltpbutton_refresh)
        self.ppbutton.clicked.connect(self.lppbutton_refresh)
        self.fftbutton.clicked.connect(self.lfftbutton_refresh)

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

    def rtpbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def rppbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticPplotRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def rfftbutton_refresh(self):
        self.centralWidget.close()
        self.layout.removeWidget(self.sc)
        self.sc = StaticFFTplotRCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.centralWidget.close()

    def draw_Rcanvas(self):
        self.centralWidget.close()
        self.centralWidget = QtGui.QWidget(self)
        self.tabs = QtGui.QTabWidget(self.centralWidget)
        self.tab1 = QtGui.QWidget(self.tabs)
        self.tpbutton = QtGui.QPushButton('Time Plot', self.tabs)
        self.tpbutton.setToolTip('Generate a plot of the membrane potential over time')
        self.ppbutton = QtGui.QPushButton('Phase Plot', self.tabs)
        self.ppbutton.setToolTip('Generate a phase plot for the system')
        self.fftbutton = QtGui.QPushButton('Power Spectrum', self.tabs)
        self.fftbutton.setToolTip('Generate the power spectrum for the signal')
        self.tab2 = QtGui.QWidget(self.tabs)
        self.tab3 = QtGui.QWidget(self.tabs)

        self.layout = QtGui.QVBoxLayout(self.tab1)
        self.hbox = QtGui.QHBoxLayout(self.tab1)
        self.layout.addLayout(self.hbox)

        self.sc = StaticNullCanvas(self.tab1, width=7, height=7, dpi=70)
        self.layout.addWidget(self.sc)
        self.hbox.addWidget(self.tpbutton)
        self.hbox.addWidget(self.ppbutton)
        self.hbox.addWidget(self.fftbutton)

        self.tpbutton.clicked.connect(self.rtpbutton_refresh)
        self.ppbutton.clicked.connect(self.rppbutton_refresh)
        self.fftbutton.clicked.connect(self.rfftbutton_refresh)

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

        Supported models are the van der Pol oscillator,
        Fitzhugh-Nagumo, Morris-Lecar, Izikevich,
        Hindmarsh-Rose and Hodgkins-Huxley, the Rikitake
        Dynamo, the Lorenz Equations and the Robbins
        Model.
        """)

    def copyright(self):
        QtGui.QMessageBox.about(self, "Copyright",
        """Copyright (C) 2015 by Zechariah Thurman
        GNU GPLv2

        This program is free software; you can redistribute
        it and/or modify it under the terms of the GNU
        General Public License as published by the Free
        Software Foundation; either version 2 of the License,
        or (at your option) any later version.

        This program is distributed in the hope that it will be
        useful, but WITHOUT ANY WARRANTY; without even
        the implied warranty of MERCHANTABILITY or FITNESS
        FOR A PARTICULAR PURPOSE. See the GNU General
        Public License for more details.

        You should have received a copy of the GNU General
        Public License along with this program; if not,
        write to the Free Software Foundation, Inc.,

        51 Franklin Street, Fifth Floor,
        Boston, MA  02110-1301, USA.
        """
        )

if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('NeuroFizzMath' + ' ' + progversion)
    aw.show()
    sys.exit(qApp.exec_())
