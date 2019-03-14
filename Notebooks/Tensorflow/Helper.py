# Please reference Documentation.ipynb for more details

# Shebang
import tensorflow as tf
import numpy as np
from numpy.fft import fft, fftfreq, rfft, fftshift
import matplotlib.pyplot as plt
import logging

#  Using this to circumvent some borked sauces in the tensorflow 1.7.0 version that I'm on in combination with
# Arch and python 3.6


class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg
        return not tf_warning


logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

# Generate solutions to the ODEs, solve up that ferndangled biscuit


def generate_odesolution(function, initial_conditions, t0=0, tfinal=50, n=1000):
    init_state = tf.constant(initial_conditions, dtype=tf.float64)
    t = np.linspace(t0, tfinal, num=n)
    tensor_state, tensor_info = tf.contrib.integrate.odeint(function, init_state, t, full_output=True)
    return [tensor_state, tensor_info]

# Function that removes the DC component of a signal so that power spectrum doesn't have peak at 0


def dc_remover(membranepotential):
    dc_component = np.mean(membranepotential) # determine DC component of membrane potential signal
    signal_sans_dc = membranepotential - dc_component # subtract DC component from PS to get rid of peak at 0
    return signal_sans_dc

# Generate a power spectrum, gotta do some twerking to get the shpongle fongled


def generate_powerspectrum(membranepotential, tfinal, n):
    X = dc_remover(membranepotential)
    fdata = X.size
    ps = abs(fft(X))**2
    time_step = tfinal/n
    freqs = fftfreq(fdata, time_step)
    idx = np.argsort(freqs)
    return freqs, ps, idx

# Generate the max frequency of the membrane potential at a given input


def generate_maxfrequency(membranepotential, tfinal, n):
    X = dc_remover(membranepotential)
    frame_rate = tfinal/10
    fdata = X.size
    ps = abs(fft(X))**2
    time_step = tfinal/n
    freqs = fftfreq(fdata)
    idx = np.argmax(np.abs(ps))
    maxfreq = np.abs(freqs[idx])
    maxfreq = np.abs(maxfreq*frame_rate)
    return maxfreq

# Plotting helper functions


def plotface(arg1, arg2=None, xlim=None, ylim=None, xlabel=None, ylabel=None, grid=None):
    plt.figure()
    if arg2 is not None:
        plt.plot(arg1, arg2)
    else:
        plt.plot(arg1)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if grid is not None:
        plt.grid()
    return plt.show()
