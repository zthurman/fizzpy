#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzTimer
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
from NeuroFizzModel import VDP, FN, ML, IZ, HR, HH
from NeuroFizzSolver import euler, ord2, rk4
import time as tm

# Time execution of the different models evaluated against different solvers

# Euler Solver

starttime = tm.time()
eqns = VDP()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for VDP against euler" % (tm.time() - starttime))

starttime = tm.time()
eqns = FN()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for FN against euler" % (tm.time() - starttime))

starttime = tm.time()
eqns = ML()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for ML against euler" % (tm.time() - starttime))

starttime = tm.time()
eqns = IZ()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for IZ against euler" % (tm.time() - starttime))

starttime = tm.time()
eqns = HR()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HR against euler" % (tm.time() - starttime))

starttime = tm.time()
eqns = HH()
solvedmodel = euler(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HH against euler" % (tm.time() - starttime))

# Second Order Runge-Kutte Solver

starttime = tm.time()
eqns = VDP()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for VDP against ord2" % (tm.time() - starttime))

starttime = tm.time()
eqns = FN()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for FN against ord2" % (tm.time() - starttime))

starttime = tm.time()
eqns = ML()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for ML against ord2" % (tm.time() - starttime))

starttime = tm.time()
eqns = IZ()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for IZ against ord2" % (tm.time() - starttime))

starttime = tm.time()
eqns = HR()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HR against ord2" % (tm.time() - starttime))

starttime = tm.time()
eqns = HH()
solvedmodel = ord2(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HH against ord2" % (tm.time() - starttime))

# Fourth Order Runge-Kutte

starttime = tm.time()
eqns = VDP()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for VDP against rk4" % (tm.time() - starttime))

starttime = tm.time()
eqns = FN()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for FN against rk4" % (tm.time() - starttime))

starttime = tm.time()
eqns = ML()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for ML against rk4" % (tm.time() - starttime))

starttime = tm.time()
eqns = IZ()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for IZ against rk4" % (tm.time() - starttime))

starttime = tm.time()
eqns = HR()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HR against rk4" % (tm.time() - starttime))

starttime = tm.time()
eqns = HH()
solvedmodel = rk4(eqns.name, eqns.xaxis, eqns.yaxis, eqns.x0, eqns.dt, eqns.t_array, eqns.eqns)
membranepotential = solvedmodel.evaluate()
print(" %s seconds for HH against rk4" % (tm.time() - starttime))