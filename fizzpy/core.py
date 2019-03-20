import tensorflow as tf
import numpy as np


class Model:

    """ Model class
    Used for storing methods related to the solution of the individual models.
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
        initial_time : int
            Initial time value for the time array.
        final_time : int
            Final time value for the time array.
        time_steps : int
            Number of steps between initial and final time in time array.

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
        initial_time : int
            Initial time value for the time array.
        final_time : int
            Final time value for the time array.
        time_steps : int
            Number of steps between initial and final time in time array.

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


class DampedSHM(Model):

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

    def solve(self):
        self.solution = self.tf_session(self.equations, self.initial_conditions)
        return self.solution


class CoupledDampedSHM(Model):

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

    def solve(self):
        self.solution = self.tf_session(self.equations, self.initial_conditions)
        return self.solution


class Lorenz(Model):

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

    def equations(self, state, t):
        x, y, z = tf.unstack(state)
        dx = self.model_parameters[1] * (y - x)
        dy = x * (self.model_parameters[0] - z) - y
        dz = x * y - self.model_parameters[2] * z
        return tf.stack([dx, dy, dz])

    def solve(self):
        self.solution = self.tf_session(self.equations, self.initial_conditions)
        return self.solution