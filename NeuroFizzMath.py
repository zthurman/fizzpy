#!/usr/bin/env python
#  Eventually I'd like to create an application that provides a nice intuitive GUI that enables the user to #manipulate different models of differential equations to exhibit different behaviors given different input #parameters.

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt

class Neuron:
	params = []
	def model(eqns,params):
		
