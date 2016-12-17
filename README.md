# NeuroFizzMath

Welcome to NeuroFizzMath, a python library for numerically solving various differential equations.

## Intended Functionality
* This library is meant to be used for generating solution arrays for differential equations
* The base of the project is NeuroFizzMath.py and the solutionGenerator method therein
* Valid model inputs at this time are:  
    -'VDP', for the Van Der Pol oscillator  
    -'LIF', for the Leaky Integrate and Fire neuron model  
    -'FN', for the Fitzhugh-Nagumo neuron model  
    -'ML', for the Morris-Lecar neuron model  
    -'IZ', for the Izhikevich neuron model  
    -'HR', for the Hindmarsh-Rose neuron model  
    -'HH', for the Hodgkin-Huxley neuron model   
* Valid solver inputs at this time are:  
    -'euler', for a simple numerical solver using Euler's method  
    -'ord2', for a second order Euler's method solver  
    -'rk4', for a fourth order Runge-Kutte solver  
* Example syntax for solving the Hodgkin-Huxley model with default parameters using Runge-Kutte:  
    solutionArray = solutionGenerator('HH', 'rk4')
* Stock project plotter can be used to generate time plot, phase plot or power spectrum, or can feed solution arrays into a custom plotter

## Punch List/Project Lifecycle Roadmap
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
* [ ] Resolve all non-abandoned or non-asserted status known issues
* [ ] Analyze execution speed for all models
* [ ] Figure out how to enable evaluating models with non-default parameters
* [ ] Figure out how to do coupled models in anything remotely resembling an elegant way
* [ ] Figure out how to introduce noise to both coupled and uncoupled models for study of stochastic and coherence resonance
* [ ] Figure out how to OOP all this sh*t
* [ ] Add approximately a bajillion other models
* [ ] Do it all in c++ and Fortran to learn those languages and see what kind of speed improvement might be gained
* [ ] Write a paper outlining the project in its painful technical details and do a thorough speed analysis

## Known Issues
Every effort will be made to keep the items on this list as few as possible with additional information regarding the issue status.

Issue | Status
------------ | -------------
The Leaky Integrate and Fire model is borked, should be faster.  | In Progress
Plotter isn't working.                                           | Resolved
HH membrane potential needs to be multiplied by -1 to be plotted.| Assessing Implications
There's some gnarlymaths in this project, there will be bugs.    | Assertion
The PyQT GUI is borked beyond repair, dont use it.               | Abandoned

## Copyright:

NeuroFizzMath - Numerical Differential Equation Solving Toolkit

Copyright (C) 2016  Zechariah Thurman

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.