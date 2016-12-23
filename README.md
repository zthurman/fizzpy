# FizzPyX

Welcome to FizzPyX, a library for numerically solving various differential equations. This library
has basically become a sandbox for learning the practice of programming, the principles of high 
performance scientific computing as well as solving as many differential equations as possible in 
one place.

## Package Requirements

* This library is being built for use on Linux as a first priority, once that is complete porting
over to OSX and Windows will be evaluated
* Base project requirements are NumPy, Math and Matplotlib, some variation of the project with just these
simple requirements will always exist for simplicity's sake
* For the more adventurous among us there will also be the more facerollonkeyboard llvmlite/Numba implementation
in an attempt to break all traditional pythonic speed barriers
    -If using PyCharm create a Conda environment  
    -Install llvmlite  
    -Install Numba  
    
## Intended Functionality
* This library is meant to be used for generating solution arrays for differential equations
* The FizzPyXScipy file has all of the model ready to be solved SciPy fashion 
* For the more adventurous, the base of the DIY project is FizzPyX.py and the solutionGenerator function therein
* Note, modelDictionary and solverDictionary are the master lists of models with their dimensions and solvers
* Valid model inputs at this time are:  
    -'VDP', for the Van Der Pol oscillator  
    -'SHM', for damped simple harmonic motion  
    -'LIF', for the Leaky Integrate and Fire neuron model  
    -'FN', for the Fitzhugh-Nagumo neuron model  
    -'ML', for the Morris-Lecar neuron model  
    -'IZ', for the Izhikevich neuron model  
    -'HR', for the Hindmarsh-Rose neuron model  
    -'RB', for the Robbins model for geomagnetic polarity reversal  
    -'LO', for the Lorenz atmospheric model, strange attractor  
    -'HH', for the Hodgkin-Huxley neuron model  
    -'CO', for a model of two coupled oscillators  
    -'RI', for the Rikitake Dynamo model for geomagnetic polarity reversal 
* Valid solver inputs at this time are:  
    -'euler', for a simple numerical solver using Euler's method  
    -'ord2', for a second order Euler's method solver  
    -'rk4', for a fourth order Runge-Kutta solver  
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
* [x] Try to extend collegiate senior project to more than one neuron model
* [x] Determine that this is neato and that this should indeed be a thing
* [x] Convert MatLab code to Python because $3000 is a big license fee after graduation and free is nice
* [x] Realize that one haz no idea what one is doing and bother programmer friends for guidance
* [x] Have friends take pity on self for self's woeful ignorance 
* [x] Get basic stuff working with help, then move everything into first Github repo evaaaaar
* [x] Determine that all work done so far is silly and that GUI is required
* [x] Create new Github repo
* [x] Try building a GUI with PyQT
* [x] Determine that GUI's are stupid and backend projects manipulating data arrays are more fun
* [x] Because so much work was done on GUI, don't delete code just let it fade into obscurity in Archive directory
* [x] Give OOP and modularity a try all at once after getting mad at PyQT and fail miserably
* [x] Ragequit ┻━┻ ︵ヽ(`Д´)ﾉ︵﻿ ┻━┻
* [x] Let project sit for a year and gently fester and putrify at the back of mind while immersing self in new employment
* [x] Come back thinking that the conceptual ideology behind of the practice of programming is now grasped 
* [x] Learn how to utilize modularity in project
* [x] Make silly punchlist in the project Readme so that self can track progress
* [x] Resolve all non-abandoned or non-asserted status known issues
* [x] Analyze execution speed for all models
* [x] Learn how to utilize modularity in project again, because adding a new model in the old structure was a hassle
* [x] Just let the archive directory go, there will always be someone to make a GUI
* [x] Figure out how to enable evaluating solvers with non-default parameters
* [ ] Do speed analysis against SciPy solver with same models tested against each
* [ ] Figure out how to enable evaluating models with non-default model specific parameters
* [ ] Figure out how to do coupled models in anything remotely resembling an elegant way
* [ ] Figure out how to introduce noise to both coupled and uncoupled models for study of stochastic and coherence resonance
* [ ] Figure out how to OO all this sh*t
* [ ] Add approximately a bajillion other models
* [ ] Do it all in Fortran 
* [ ] Do it all in c++
* [ ] Write a paper outlining the project in its painful technical details and do a thorough speed analysis