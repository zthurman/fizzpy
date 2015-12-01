NeuroFizzMath
=============

Computational neuroscience and AI simulation toolkit.

Goal
=============

The goal of developing this library is to provide an open source reference for computational neuroscience and artificial intelligence frameworks. Primarily how we can use the one to
help make inspire realistic efficiency assessments and optimizations to the other.

Things to Note:
=============

This is a work in progress, as such it is not finished yet.

The application is licensed under the GNU GPLv2. As such it is copylefted and meant for free use, modification and re-distribution subject to the terms of the Free Software Foundation,
Inc.

User Manual:
=============

This library is meant to provide a backend to some sort of UI. Presently PyQt has been the choice for the UI side of things but yours truly find himself heavily grossed out by the massive
undertaking of writing a UI from scratch with the PyQt framework. As such this library is in flux for a non-programmer end user. However, if you're comfortable with manipulating numpy
arrays and want to do your own thing with the raw data all that you need are NeuroFizzModel and NeuroFizzSolver.

NeuroFizzModel is made up of different systems of differential equations all packaged up into classes, in and of themselves these aren't particularly useful but when you invoke one of the
solvers within NeuroFizzSolver and plug one of the models in from NeuroFizzModel an array of numerically integrated data is generated.

Reference NeuroFizzPlot for a simple example of how to use NeuroFizzModel and NeuroFizzSolver.

Requirements:
=============
-Refer to requirements.txt for the venv dependencies