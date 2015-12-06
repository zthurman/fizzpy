#!/usr/bin/env python
# NeuroFizzMath - NeuroFizzSpikes
# Copyright (C) 2015 Zechariah Thurman
# GNU GPLv2

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math as mt

# Generating a Poisson spike train with refractoriness

fr_mean = 15/1000
lam = 1/fr_mean
ns = 1000
isi = []
isi1 = -lam*np.log(np.random(ns, 1))
isi2 = 0
rand = np.random

for i in np.arange(0, ns):
    if rand > mt.exp(-isi1(i)**(2/32)):
        isi2 += 1
        isi(isi2) = isi1(i)

plt.hist(isi, 50)





