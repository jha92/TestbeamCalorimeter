#! /usr/bin/env python
'''plot calibration of scope data

Usage:
    plot_calibration.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
name = 'Air'
color_name = 'deepskyblue'

# DATA
energy,volt,sigma = np.loadtxt('fitparameter.txt', delimiter=',', usecols=(0,1,2), unpack=True)
energy_spread = 0.158

# FIT linear
fitfunc = lambda x, *p: p[0] + p[1] * x
xdata = volt
ydata = energy
dydata = sigma; dydata = np.where(dydata > 0.0, dydata, 1)
para, cov = curve_fit(fitfunc, xdata, ydata, p0=[0.,1.], sigma=dydata)
perr = np.sqrt(np.diag(cov))

##########
props = dict(boxstyle='square', facecolor='white')

### plot calibration curve
fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
plt.errorbar(volt, energy, xerr=sigma, yerr=energy_spread, marker='o', markersize=2, linestyle='None', color=color_name)
data_info = ('Calibration with ' + name)
plt.text(0.05, 0.95, data_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props)
# options
plt.ylim(0., 4.5)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'beam energy [GeV]')
plt.xlim(0.,4.)
# save
plt.savefig("./plots/calibration.png")


### plot calibration curve with fit
fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.errorbar(volt, energy, xerr=sigma, yerr=energy_spread, marker='o', markersize=2, linestyle='None', color=color_name, elinewidth=0.8)
data_info = ('Calibration with ' + name)
plt.text(0.05, 0.95, data_info, fontsize=10, transform = ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)
fit_info = (r'offset$=(%.2f\pm%.2f)$GeV, $c=(%.2f\pm%.2f)\frac{GeV}{V}$'%(para[0],perr[0],para[1],perr[1]))
plt.text(0.15, 0.1, fit_info, fontsize=10, transform = ax.transAxes, verticalalignment='top', horizontalalignment='left')
fit_info = (r'E[GeV]$ = c \cdot $pulse[V]$ + $offset')
plt.text(0.45, 0.4, fit_info, fontsize=10, transform = ax.transAxes, verticalalignment='top', horizontalalignment='left')
x_fit = np.arange(0,4.5)
y_fit = fitfunc(x_fit, *para)
plt.plot(x_fit, y_fit, ls='-', lw=1, alpha=0.8, color='red')
# options
plt.ylim(0., 4.5)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'beam energy [GeV]')
plt.xlim(0.,4.5)
# save
plt.savefig("./plots/calibration_fit.png")
