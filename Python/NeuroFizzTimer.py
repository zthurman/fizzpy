#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzTimer
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from time import time
from Python.NeuroFizzMath import solutionGenerator


def solutionTimer(numberoftimestorun, modelname, solvername):
    i = 0
    total = 0
    while i < numberoftimestorun:
        start = time()
        solutionGenerator(modelname, solvername)
        end = time()
        elapsed = end - start
        total += elapsed
        i += 1
    return total, numberoftimestorun


def solutiontimeAverager(total, numberoftimestorun):
    return total/numberoftimestorun


def solutiontimeAggregator():
    pass


if __name__ == '__main__':
    times = solutionTimer(100, 'IZ', 'rk4')
    average = solutiontimeAverager(times[0], times[1])
    times1 = solutionTimer(100, 'LIF', 'rk4')
    average1 = solutiontimeAverager(times1[0], times1[1])
    times2 = solutionTimer(100, 'FN', 'rk4')
    average2 = solutiontimeAverager(times2[0], times2[1])
    print(average)
    print(average1)
    print(average2)

