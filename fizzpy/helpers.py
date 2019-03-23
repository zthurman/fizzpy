import matplotlib.pyplot as plt


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
