import numpy as np
import tensorflow as tf


class Model:

    """Model class
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

    def solve(self):
        """Solve

        Solves the provided equations in a Tensorflow session with either the provided
        or the default initial conditions.

        Parameters
        ----------
        self
            Current instance state.

        Returns
        -------
        np.ndarray
            Returns the solution from the Tensorflow session.

        """
        self.solution = self.tf_session(self.equations, self.initial_conditions)
        return self.solution


class CoupledDampedSHM(Model):

    """Coupled Damped Simple Harmonic Motion

    This system of ODEs models coupled damped simple harmonic motion, such as two carts
    on a track coupled to each other and each edge of the track by springs.

    """

    def __init__(
            self,
            initial_conditions=[0.5, 0.1, 0.1, 0.1],
            model_parameters=[0.007, 0.27, 0.027, 0.25],
            final_time=200,
            time_steps=1000):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        x, y, x1, y1 = tf.unstack(state)
        dx = y
        dy = -(self.model_parameters[1] / self.model_parameters[3]) * x \
            + (self.model_parameters[2] / self.model_parameters[3]) * x1 \
            - (self.model_parameters[0] / self.model_parameters[3]) * y
        dx1 = y1
        dy1 = (self.model_parameters[2] / self.model_parameters[3]) * x \
            - (self.model_parameters[1] / self.model_parameters[3]) * x1 \
            - (self.model_parameters[0] / self.model_parameters[3]) * y1
        return tf.stack([dx, dy, dx1, dy1])


class DampedSHM(Model):

    """Damped Simple Harmonic Motion

    This system of ODEs models damped simple harmonic motion.

    """

    def __init__(
            self,
            initial_conditions=[0.1, 0.1],
            model_parameters=[0.035, 0.5, 0.2],
            final_time=50,
            time_steps=500):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        x, y = tf.unstack(state)
        dx = y
        dy = (-self.model_parameters[0] * y - self.model_parameters[1] * x) / self.model_parameters[2]
        return tf.stack([dx, dy])


class FitzhughNagumo(Model):

    """Fitzhugh-Nagumo neuron model

    This system of ODEs is an implementation of the Fitzhugh-Nagumo
    model for the action potential of a point neuron.

    """

    def __init__(
            self,
            initial_conditions=[0.01, 0.01],
            model_parameters=[0.75, 0.8, 3, -0.4],
            final_time=100,
            time_steps=500):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        v, w = tf.unstack(state)
        dv = self.model_parameters[2] * (v + w - (v**3/3) + self.model_parameters[3])
        dw = -1/self.model_parameters[2] * (v - self.model_parameters[0] + self.model_parameters[1]*w)
        return tf.stack([dv, dw])


class HindmarshRose(Model):

    """Hindmarsh-Rose neuron model

    This system of ODEs is an implementation of the Hindmarsh-Rose
    model for the action potential of a point neuron.

    """

    def __init__(
            self,
            initial_conditions=[0.1, 0.1, 0.1],
            model_parameters=[1., 3., 1., 5., 0.006, 4., 1.3, -1.5],
            final_time=100,
            time_steps=1000):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        x, y, z = tf.unstack(state)
        dx = y - self.model_parameters[0] * (x ** 3) \
            + (self.model_parameters[1] * (x ** 2)) - z + self.model_parameters[6]
        dy = self.model_parameters[2] - self.model_parameters[3] * (x ** 2) - y
        dz = self.model_parameters[4] * (self.model_parameters[5] * (x - self.model_parameters[7]) - z)
        return tf.stack([dx, dy, dz])


class HodgkinHuxley(Model):

    """Hodgkin-Huxley neuron model

    This system of ODEs is an implementation of the Hodgkin-Huxley
    model for the action potential of a point neuron.

    """

    def __init__(
            self,
            initial_conditions=[0.1, 0.1, 0.1, 0.1],
            model_parameters=[36., 120., 0.3, 12., -115., -10.613, 1., -10.],
            final_time=100,
            time_steps=1000):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        i, n, m, h = tf.unstack(state)
        # Alpha and beta functions for channel activation functions
        alpha_n = (0.01 * (i + 10)) / (tf.exp((i + 10) / 10) - 1)
        beta_n = 0.125 * tf.exp(i / 80)
        alpha_m = (0.1 * (i + 25)) / (tf.exp((i + 25) / 10) - 1)
        beta_m = 4 * tf.exp(i / 18)
        alpha_h = (0.07 * tf.exp(i / 20))
        beta_h = 1 / (tf.exp((i + 30) / 10) + 1)
        # Differential Equations
        di = (self.model_parameters[0] * (n ** 4) * (i - self.model_parameters[3])
              + self.model_parameters[1] * (m ** 3) * h * (i - self.model_parameters[4])
              + self.model_parameters[2] * (i - self.model_parameters[5])
              - self.model_parameters[7]) * (-1 / self.model_parameters[6])
        dn = alpha_n * (1 - n) - beta_n * n
        dm = alpha_m * (1 - m) - beta_m * m
        dh = alpha_h * (1 - h) - beta_h * h
        return tf.stack([di, dn, dm, dh])

    def solve(self):
        i, n, m, h = self.tf_session(self.equations, self.initial_conditions)
        self.solution = -1*i, n, m, h
        return self.solution


class HIV(Model):

    """HIV dynamics

    This system of ODEs is an implementation of a model for HIV
    dynamics in a T-cell population.

    """

    def __init__(
            self,
            initial_conditions=[1000, 0, 1],
            model_parameters=[10., 0.02, 0.24, 2.4, 2.4e-5, 100],
            final_time=500,
            time_steps=500):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        x1, x2, x3 = tf.unstack(state)
        dx1 = -self.model_parameters[1] * x1 - self.model_parameters[4] * x1 * x3 + self.model_parameters[0]
        dx2 = -self.model_parameters[3] * x2 + self.model_parameters[4] * x1 * x3
        dx3 = self.model_parameters[5] * x2 - self.model_parameters[2] * x3
        return tf.stack([dx1, dx2, dx3])


class Lorenz(Model):

    """Lorenz equations

    This system of ODEs is an implementation of the Lorenz equations
    which model atmospheric convection.

    """

    def __init__(
            self,
            initial_conditions=[0, 2, 20],
            model_parameters=[28., 10., 8. / 3.],
            final_time=50,
            time_steps=5000):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps
        self.state = tf.tensor()

    def equations(self, state, t):
        x, y, z = tf.unstack(state)
        dx = self.model_parameters[1] * (y - x)
        dy = x * (self.model_parameters[0] - z) - y
        dz = x * y - self.model_parameters[2] * z
        return tf.stack([dx, dy, dz])


class MorrisLecar(Model):

    """Morris-Lecar neuron model

    This system of ODEs is an implementation of the Morris-Lecar
    model for the action potential of a point neuron.

    """

    def __init__(
            self,
            initial_conditions=[0.01, 0.01],
            model_parameters=[-84., 8., 130., 4.4, -60., 2., 0.04, -1.2, 18., 2., 30., 80.],
            final_time=500,
            time_steps=1000):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        v, n = tf.unstack(state)
        dv = (-self.model_parameters[3]
              * (0.5 * (1 + tf.tanh((v - self.model_parameters[7]) / self.model_parameters[8])))
              * (v - self.model_parameters[2]) - self.model_parameters[1] * n
              * (v - self.model_parameters[0]) - self.model_parameters[5]
              * (v - self.model_parameters[4]) + self.model_parameters[11])
        dn = (self.model_parameters[6]
              * ((0.5 * (1 + tf.tanh((v - self.model_parameters[9]) / self.model_parameters[10]))) - n)) \
            / (1 / tf.cosh((v - self.model_parameters[9]) / (2 * self.model_parameters[10])))
        return tf.stack([dv, dn])


class Vanderpol(Model):

    """Van der pol oscillator

    This system of ODEs is an implementation of the van der pol
    oscillator, a commonly used introductory system in the study
    of dynamical systems.

    """

    def __init__(
            self,
            initial_conditions=[0.01, 0.01],
            model_parameters=[-0.05],
            final_time=50,
            time_steps=250):
        self.initial_conditions = np.array(initial_conditions)
        self.model_parameters = model_parameters
        self.final_time = final_time
        self.time_steps = time_steps

    def equations(self, state, t):
        x, y = tf.unstack(state)
        dx = y
        dy = self.model_parameters[0]*y*(1 - x**2) - x
        return tf.stack([dx, dy])
