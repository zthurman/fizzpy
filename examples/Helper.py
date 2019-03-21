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


# Generate Tensorflow session


def generate_tensorflowsession(function, initial_conditions, t0=0, tfinal=50, n=1000):
    sess = tf.Session()
    state, info = sess.run(generate_odesolution(function, initial_conditions, t0=t0, tfinal=tfinal, n=n))
    columns = len(state.T)
    rows = len(state)
    output = np.ndarray(columns)
    output = state.T
    return output


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


class Model:

    """ Model class
    Used for storing methods that generalize to all models.
    """

    def __init__(
            self,
            initial_conditions=None,
            model_parameters=None,
            final_time=None,
            time_steps=None):
        self.initial_conditions = initial_conditions
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def init_converter(self, arg1: np.array) -> tf.constant:
        """Initial conditions converter.

        Converts the initial_conditions ndarray into a Tensorflow
        constant n-dimensional tensor.

        `tf.constant <https://www.tensorflow.org/api_docs/python/tf/constant>`_

        Parameters
        ----------
        arg1
            Initial conditions for the system of ODEs to be solved.

        Returns
        -------
        tf.constant
            Constant tf.float64 n-d tensor based on initial_conditions
            provided.

        """
        init_state = tf.constant(arg1, dtype=tf.float64)
        return init_state

    def ode_solver(self, arg1: tf.stack, arg2: np.ndarray) -> list:
        """Ordinary Differential Equation (ODE) solver.

        Uses Tensorflow/Numpy odeint to numerically solve the input system
        of ODEs given the provided initial conditions and optional time
        array parameters.

        `odeint <https://www.tensorflow.org/api_docs/python/tf/contrib/integrate/odeint>`_

        Parameters
        ----------
        arg1
            Tensorflow stack representing the equations for the system of ODEs.
        arg2
            Initial conditions for the system of ODEs to be solved.

        Returns
        -------
        list
            y: (n+1)-d tensor. Contains the solved value of y for each desired
            time point in t.
            info_dict: only if full_output=True for odeint, additional info.

        """
        t = np.linspace(0, self.final_time, num=self.time_steps)
        tensor_state, tensor_info = tf.contrib.integrate.odeint(arg1, self.init_converter(arg2), t, full_output=True)
        return [tensor_state, tensor_info]

    def tf_session(self, arg1: tf.stack, arg2: np.ndarray) -> np.ndarray:
        """Tensorflow session runner.

        Uses a Tensorflow session run to evaluate the provided system of ODEs.

        `tf.Session.run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_

        Parameters
        ----------
        arg1
            Tensorflow stack representing the equations for the system of ODEs.
        arg2
            Initial conditions for the system of ODEs to be solved.

        Returns
        -------
        np.ndarray
            Returns the transpose of the (n+1)-d state tensor returned from
            ode_solver after it's been solved in the Tensorflow session.

        """
        sess = tf.Session()
        state, info = sess.run(self.ode_solver(arg1, arg2))
        output = state.T
        return output
