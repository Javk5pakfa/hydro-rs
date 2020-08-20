#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('output.dat')

x, rho, vel, pre = data.T

plt.plot(x, rho, '-o', mfc='none')
plt.show()
