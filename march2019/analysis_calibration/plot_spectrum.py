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
name = 'Air'
color_name = 'deepskyblue'

data_inputs = ['../data/190308_122427.csv','../data/190308_122947.csv','../data/190308_123553.csv','../data/190308_124247.csv','../data/190308_125516.csv']
data_names = ['1GeV','2GeV','3GeV','4GeV','5GeV']

for i in range(len(data_inputs)):

	# DATA
	with open(data_inputs[i],'r') as infile:
	    data = infile.readlines()[1:-1]
	bins  = np.array([float(row.split(",")[0]) for row in data])
	counts = np.array([float(row.split(",")[1]) for row in data])
	bins = bins * -1.
	binning = bins[1] - bins[0]
	bins = bins - binning/2
	counts = counts / 637

	# FIT / Mean
	fitfunc = lambda x, *p: p[2] * np.exp(-0.5*(x-p[0])**2/p[1]**2)
	maximum_bin_index = np.where(np.max(counts)==counts)[0][0]
	mu_guess = bins[maximum_bin_index]
	si_guess = 1.
	norm_guess = np.max(counts)
	para0 = [mu_guess, si_guess, norm_guess] # mu, sigma, norm
	# fit 
	index_around = 30
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
	total_counts = np.sum(counts)


	# PLOT
	props = dict(boxstyle='square', facecolor='white')
	bin_low = 0
	bin_up = 5.5

	### spectral distribution
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins, counts, width=binning, color=color_name, label='%s'%data_names[i])
	plt.yscale('log')
	plt.ylim(1., 1e4)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'counts [#]')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/%s_spectrum.png'%data_names[i])


	### spectral distribution with fit
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins, counts, width=binning, color=color_name, label='%s'%data_names[i])
	fit_info = (r'total counts = %1d'%(total_counts) + '\n' +
        	    r'$\mu = %.2f\pm%.2f$'%(para[0],perr[0]) + '\n' +
            	    r'$\sigma = %.2f\pm%.2f$'%(abs(para[1]),perr[1])
		    )
	plt.text(0.05, 0.95, fit_info, fontsize=8, transform = ax.transAxes, verticalalignment='top', horizontalalignment='left')
	x_fit = xdata
	y_fit = fitfunc(x_fit, *para)
	plt.plot(x_fit, y_fit, ls='--', lw=1, alpha=0.8, color='red')
	plt.yscale('log')
	plt.ylim(1., 1e4)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'counts [#]')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/%s_spectrum_fit.png'%data_names[i])
	with open('fitparameter.txt','a') as outfile:
		outfile.write('%s,%.4f,%.4f\n'%(data_names[i],para[0],abs(para[1])))


	### normalized spectral distribution
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	counts_norm = counts/total_counts
	plt.bar(bins, counts_norm, width=binning, color=color_name, label='%s'%data_names[i])
	plt.yscale('log')
	plt.ylim(1e-4, 1e0)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/%s_spectrum_normalized.png'%data_names[i])


	### normalized spectral distribution with fit
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins, counts_norm, width=binning, color=color_name, label='%s'%data_names[i])
	fit_info = (r'$\mu = %.2f\pm%.2f$'%(para[0],perr[0]) + '\n' +
            	    r'$\sigma = %.2f\pm%.2f$'%(abs(para[1]),perr[1])
		    )
	plt.text(0.05, 0.95, fit_info, fontsize=8, transform = ax.transAxes, verticalalignment='top', horizontalalignment='left')

	plt.plot(x_fit, y_fit/total_counts, ls='--', lw=1, alpha=0.8, color='red')
	plt.yscale('log')
	plt.ylim(1e-4, 1e0)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/%s_spectrum_normalized_fit.png'%data_names[i])



