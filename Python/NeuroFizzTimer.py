#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzTimer
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from time import time
from Python.NeuroFizzMath import solutionGenerator, modelDictionary, getModelDictionaryKeys, solverList


# Time performance evaluator function

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
    return total, numberoftimestorun, modelname, solvername


# Averager function

def solutiontimeAverager(total, numberoftimestorun):
    return total/numberoftimestorun


# Aggregates time performance data for all models with all solvers

def solutiontimeAggregator(numberoftimestorun):
    averages = []
    for i in getModelDictionaryKeys(modelDictionary):
        for j in solverList:
            solutiontime = solutionTimer(numberoftimestorun, i, j)
            average = solutiontimeAverager(solutiontime[0], solutiontime[1])
            averages.append([average, solutiontime[2], solutiontime[3]])
    return averages


# Main

if __name__ == '__main__':
    # Be patient, the higher the number of times to run the more you wait
    performanceevaluation = solutiontimeAggregator(1)
    print(performanceevaluation)


