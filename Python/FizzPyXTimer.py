#!/usr/bin/env python
# FizzPyX - FizzPyXTimer
# Copyright (C) 2016 Zechariah Thurman
# GNU GPLv3


from __future__ import division
from time import time
from numpy import array

from Python.FizzPyX import solutionGenerator, modelDictionary, getModelDictionaryKeys, solverList


# Time performance evaluator function

def solutionTimer(numberoftimestorun, modelname, solvername, inits=None, endtime=None, timestep=None):
    i = 0
    total = 0
    if 'inits' is not None or 'endtime' is not None or 'timestep' is not None:
        if inits is not None:
            inits = inits
        if endtime is not None:
            endtime = endtime
        if timestep is not None:
            timestep = timestep
        while i < numberoftimestorun:
            start = time()
            solutionGenerator(modelname, solvername, inits=inits, endtime=endtime, timestep=timestep)
            end = time()
            elapsed = end - start
            total += elapsed
            i += 1
        return total, numberoftimestorun, modelname, solvername
    else:
        while i < numberoftimestorun:
            start = time()
            solutionGenerator(modelname, solvername, inits=None, endtime=None, timestep=None)
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


# Aggregates time performance data for selected model with all solvers

def solutiontimeEvaluator(numberoftimestorun, modelname, inits=None, endtime=None, timestep=None):
    averages = []
    inits = inits
    endtime = endtime
    timestep = timestep
    for i in solverList:
        solutiontime = solutionTimer(numberoftimestorun, modelname, i, inits=inits, endtime=endtime, timestep=timestep)
        average = solutiontimeAverager(solutiontime[0], solutiontime[1])
        averages.append([average, solutiontime[2], solutiontime[3]])
    return averages


# Main

if __name__ == '__main__':
    # Default parameters
    performanceevaluationfn = solutiontimeEvaluator(50, 'FN')
    print(performanceevaluationfn)
    performanceevaluationiz = solutiontimeEvaluator(50, 'IZ')
    print(performanceevaluationiz)
    performanceevaluationHH = solutiontimeEvaluator(50, 'LIF')
    print(performanceevaluationHH)

    # # Non-default parameters
    # performanceevaluationfn = solutiontimeEvaluator(100, 'FN', inits=array([0.05, 0.02]), endtime=500, timestep=0.05)
    # print(performanceevaluationfn)
    # performanceevaluationiz = solutiontimeEvaluator(100, 'IZ', inits=array([0.05, 0.02]), endtime=500, timestep=0.05)
    # print(performanceevaluationiz)
    # performanceevaluationHH = solutiontimeEvaluator(100, 'HH',
    #                                                 inits=array([0.05, 0.02, 0.035, 0.01]), endtime=500, timestep=0.05)
    # print(performanceevaluationHH)

    # Be patient, this one does all of the models
    # performanceevaluation = solutiontimeAggregator(1)
    # print(performanceevaluation)
