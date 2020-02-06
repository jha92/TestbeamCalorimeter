#! /usr/bin/env python
'''plot and analyse histograms from MSO scope

Usage:
    plot_spectrum.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
data_name = '../data/air/2_0Gev_180817.csv'

# DATA
with open(data_name,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])

# make bins positive 
bins = bins * -1.
# shift bins in the center
binning = bins[1] - bins[0]
bins = bins - binning/2
# normalize counts
counts = counts / 500
counts = counts / np.sum(counts)

# FIT / Mean
fitfunc = lambda x, *p: p[2] * np.exp(-0.5*(x-p[0])**2/p[1]**2)
maximum_bin_index = np.where(np.max(counts)==counts)[0][0]
mu_guess = bins[maximum_bin_index]
si_guess = 1.
norm_guess = np.max(counts)
para0 = [mu_guess, si_guess, norm_guess] # mu, sigma, norm
# fit 
index_around = 12
start_index = maximum_bin_index-index_around
end_index = maximum_bin_index+index_around
xdata = bins[start_index:end_index]
ydata = counts[start_index:end_index]
dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1)
para, cov = curve_fit(fitfunc, xdata, ydata, p0=para0, sigma=dydata)
perr = np.sqrt(np.diag(cov))
# chi**2
chi2 = np.sum(((ydata - fitfunc(xdata, *para)) / dydata)**2)
chi2red = chi2 / (len(ydata)-len(para))


##########
props = dict(boxstyle='square', facecolor='white')
bin_low = 0
bin_up = 7


### plot normalized distribution
fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.bar(bins, counts, width=binning, color='deepskyblue', label='2GeV')
fit_info = (r'$\mu = %.2f\pm%.2f$'%(para[0],perr[0]) + '\n' +
            r'$\sigma = %.2f\pm%.2f$'%(abs(para[1]),perr[1])
	    )
plt.text(0.75, 0.95, fit_info, fontsize=10, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
x_fit = xdata
y_fit = fitfunc(x_fit, *para)
plt.plot(x_fit, y_fit, ls='--', lw=1, alpha=0.8, color='red')
plt.yscale('log')
plt.ylim(1e-4, 3e-1)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'normalized counts')
plt.xlim(0,4.5)
plt.xticks(np.arange(0,5,0.5))
plt.savefig('./plots/calibration_spectrum_2GeV.png')

