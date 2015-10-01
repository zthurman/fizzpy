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
