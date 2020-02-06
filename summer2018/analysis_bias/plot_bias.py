
import numpy as np

import matplotlib.pyplot as plt

from scipy import integrate
#import inspect, os
import scipy as sp
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import sys


# DATA
bias_2GeV, counts_2Gev, error_2GeV = np.loadtxt("../data/bias/bias_count_2_Gev.txt", usecols=(0, 1, 2), skiprows=1).T
bias_4GeV, counts_4Gev, error_4GeV = np.loadtxt("../data/bias/bias_count_4_Gev.txt", usecols=(0, 1, 2), skiprows=1).T


# PLOT for 2GeV
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
ax1.errorbar(bias_2GeV, counts_2Gev/10., yerr=error_2GeV/10., fmt='o', markersize=5, capsize=5, color='blue', label='beam momentum: 2GeV')
ax1.set_xlabel(r'Negative Bias Voltage [V]')
ax1.set_ylabel(r'Count Rate [s${}^{-1}$]')
ax1.set_xlim(1550,2050)
ax1.set_xticks(np.arange(1600, 2050, 50))
plt.legend(loc='upper left')
plt.savefig('bias_2GeV.pdf')

# PLOT for 4GeV
fig, ax1 = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
ax1.errorbar(bias_4GeV, counts_4Gev/10., yerr=error_4GeV/10., fmt='o', markersize=5, capsize=5, color='blue', label='beam momentum: 4GeV/c')
ax1.set_xlabel(r'Negative Bias Voltage [V]')
ax1.set_ylabel(r'Count Rate [s${}^{-1}$]')
ax1.set_xlim(1550,2150)
ax1.set_xticks(np.arange(1600, 2150, 50))
plt.legend(loc='upper left')
plt.savefig('bias_4GeV.pdf')

# PLOT combined
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
ax1.set_ylabel(r'Count Rate [s${}^{-1}$]', color='darkcyan')
ax1.errorbar(bias_2GeV, counts_2Gev/10., yerr=error_2GeV/10., fmt='o', markersize=5, capsize=5, color='darkcyan', label='beam momentum: 2GeV/c')
ax1.tick_params(axis='y', labelcolor='darkcyan')
ax2 = ax1.twinx()
ax2.set_ylabel(r'Count Rate [s${}^{-1}$]', color='darkblue')
ax2.errorbar(bias_4GeV, counts_4Gev/10., yerr=error_4GeV/10., fmt='o', markersize=5, capsize=5, color='darkblue', label='beam momentum: 4GeV/c')
ax2.tick_params(axis='y', labelcolor='darkblue')
ax1.set_xlim(1550,2150)
ax1.set_xticks(np.arange(1600, 2150, 50))
ax1.set_xlabel(r'Negative Bias Voltage [V]')
fig.legend(loc='upper left',bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
fig.tight_layout()
plt.savefig('bias_combined.pdf')
