NeuroFizzMath
=============

Computational neuroscience and AI simulation toolkit.

Goal
=============

The goal of developing this application is to provide an open source reference for computational neuroscience and artificial intelligence frameworks. Primarily how we can use one to
make efficiency assessments and optimizations to the other.

Things to Note:
=============

This is a work in progress, the primary focus is computational neuroscience in a referential context for AI research.

The application is licensed under the GNU GPLv2. As such it is copylefted and meant for free use, modification and re-distribution subject to the terms of the Free Software Foundation,
Inc.

User Manual:
=============

This library is meant to provide a backend to some sort of UI. Presently PyQt has been the choice for the UI side of things but yours truly find himself heavily grossed out by the massive
undertaking of writing a UI from scratch with the PyQt framework. As such this library is in flux for a non-programmer end user. However, if you're comfortable with manipulating numpy
arrays and want to do your own thing with the raw data all that you need are NeuroFizzModel and NeuroFizzSolver.

NeuroFizzModel is made up of different systems of differential equations all packaged up into classes, in and of themselves these aren't particularly useful but when you invoke one of the
solvers within NeuroFizzSolver and plug one of the models in from NeuroFizzModel an array of numerically integrated data is generated.

Pseudocode example:

    from __future__ import division
    from NeuroFizzModel import VDP
    from NeuroFizzSolver import euler

    # instantiate VDP model

    test = VDP()

    # instantiate euler solver

    solved = euler(test.name, test.x0, test.dt, test.t_array, test.eqns)

    # for plotting we care about solved.tsp and the array returned by solved.evaluate

    soln = solved.evaluate()
    soln = soln[:,0]
    t = solved.tsp

Requirements:
=============
-Refer to requirements.txt for the venv dependencies