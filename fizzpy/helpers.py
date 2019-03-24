import matplotlib.pyplot as plt


def plotface(arg1, arg2=None, xlim=None, ylim=None, xlabel=None, ylabel=None, grid=None):
    """Plot face.

    Uses matplotlib to generate a figure and plot the first argument with
    respect to the keyword arguments provided.

    `plt.plot <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_

    Parameters
    ----------
    arg1: np.array
        Solution to be plotted.
    arg2: np.array
        Optional secondary solution for the first to be plotted against,
        primarily useful for plots in phase space.
    xlim: list
        Upper and lower limits for the x-axis.
    ylim: list
        Upper and lower limits for the y-axis.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    grid: bool
        Turns the grid for the plot on or off.

    Returns
    -------
    tf.constant
        Constant tf.float64 n-d tensor based on initial_conditions
        provided.

    """
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
