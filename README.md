# FizzPyX

Welcome to FizzPyX, a library for numerically solving various differential equations with different computation methods.
This library has basically become a sandbox for learning the practice of programming, the principles of high
performance scientific computing as well as solving as many differential equations as possible in one place.

## Package Requirements

* This library is being built for use on Linux as a first priority, once that is complete porting
over to OSX and Windows will be evaluated
* Base project requirements are NumPy and Matplotlib, some variation of the project with just these
simple requirements will always exist for simplicity's sake
* Using TensorFlow to evaluate the point neurons is also on the roadmap

## Intended Functionality
* This library is meant to be used for generating solution arrays for differential equations
* The FizzPyXScipy file has all of the model ready to be solved SciPy fashion 
* For the more adventurous, the base of the DIY project is FizzPyX.py and the solutionGenerator function therein
* Note, modelDictionary and solverDictionary are the master lists of models with their dimensions and solvers
* Valid model inputs at this time are:
    *'VDP', for the Van Der Pol oscillator
    *'SHM', for damped simple harmonic motion
    *'LIF', for the Leaky Integrate and Fire neuron model
    *'FN', for the Fitzhugh-Nagumo neuron model
    *'ML', for the Morris-Lecar neuron model
    *'IZ', for the Izhikevich neuron model
    *'HR', for the Hindmarsh-Rose neuron model
    *'RB', for the Robbins model for geomagnetic polarity reversal
    *'LO', for the Lorenz attractor
    *'HH', for the Hodgkin-Huxley neuron model
    *'CO', for a model of two coupled oscillators
    *'RI', for the Rikitake Dynamo model for geomagnetic polarity reversal 
* Valid solver inputs at this time are:
    *'euler', for a simple numerical solver using Euler's method
    *'ord2', for a second order Euler's method solver
    *'rk4', for a fourth order Runge-Kutta solver  
* Example syntax for solving the Hodgkin-Huxley model with default parameters using Runge-Kutta:  
    solutionArray = solutionGenerator('HH', 'rk4')
* Example syntax for solving the Hodgkin-Huxley model with non-default parameters using Runge-Kutta:  
    solutionArray = solutionGenerator('HH', 'rk4', inits=array([0.03, 0.03]), endtime=200, timestep=0.05)
* Stock project plotter can be used to generate time plot, phase plot or power spectrum, using do_tplot,  
do_pplot or do_psplot or solution arrays can be fed into a custom plotter

## Known Issues
Every effort will be made to keep the items on this list as few as possible with additional information regarding the issue status.

Issue | Status
------------ | -------------
The Leaky Integrate and Fire model is borked, should be faster.                           | Resolved
Plotter isn't working.                                                                    | Resolved
HH membrane potential needs to be multiplied by -1 to be plotted.                         | Assertion
There's some gnarlymaths in this code, there will probably be numpy bugs if you poke it.  | Assertion
The PyQT GUI is borked beyond repair, dont use it.                                        | Abandoned

## Project Lifecycle Roadmap
* [x] Begin knowledge quest about how to understand the way a big fat bundle of interconnected point neurons give rise
to intelligence and consciousness
* [x] Ragequit ┻━┻ ︵ヽ(`Д´)ﾉ︵﻿ ┻━┻
* [x] Let project sit and gently fester for a time at the back of mind while immersing self in the world of employment
* [x] Come back with new focus and perspective
* [ ] Make shit happen

