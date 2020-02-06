#! /usr/bin/env python
'''plot and analyse histograms from MSO scope

Usage:
    plot_spectrum_Aluminum.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
name = 'Aluminum'
color_name = 'seagreen'

data_inputs = ['../data/190311_090428.csv','../data/190311_085533.csv']
beam_energies = ['2GeV','5GeV']
thickness_names = ['0pt5mm','0pt5mm']
thickness_titles = ['0.5mm','0.5mm']

for i in range(len(data_inputs)):

	# DATA
	with open(data_inputs[i],'r') as infile:
    		data = infile.readlines()[1:-1]

	bins  = np.array([float(row.split(",")[0]) for row in data])
	counts = np.array([float(row.split(",")[1]) for row in data])

	bins = bins * -1.
	binning = bins[1] - bins[0]
	bins = bins - binning/2
	counts = counts / 672
	total_counts = np.sum(counts)	
	
	##########
	props = dict(boxstyle='square', facecolor='white')
	bin_low = 0
	bin_up = 5.5

	### plot distribution
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins, counts, width=binning, color=color_name, label='%s %s @ %s'%(name,beam_energies[i],thickness_titles[i]))
	plt.yscale('log')
	plt.ylim(1., 1e4)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'counts [#]')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/spectrum_%s_%s_%s.png'%(name,beam_energies[i],thickness_names[i]))
	plt.close()	

	### plot normalized distribution
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	counts_norm = counts/total_counts
	plt.bar(bins, counts_norm, width=binning, color=color_name, label='%s %s @ %s'%(name,beam_energies[i],thickness_titles[i]))
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/spectrum_%s_%s_%s_normalized.png'%(name,beam_energies[i],thickness_names[i]))
	plt.close()		

	#FITTING
	fitfunc = lambda x, *p: p[2] * np.exp(-0.5*(x-p[0])**2/p[1]**2)
	maximum_bin_index = np.where(np.max(counts[5:])==counts)[0][0]
	mu_guess = bins[maximum_bin_index]
	si_guess = 1.
	norm_guess = np.max(counts)
	para0 = [mu_guess, si_guess, norm_guess] # mu, sigma, norm
	# fit 
	index_left = 25
	index_right = 30
	start_index = maximum_bin_index-index_left
	end_index = maximum_bin_index+index_right
	xdata = bins[start_index:end_index]
	ydata = counts[start_index:end_index]
	dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1)
	para, cov = curve_fit(fitfunc, xdata, ydata, p0=para0, sigma=dydata)
	perr = np.sqrt(np.diag(cov))
	chi2 = np.sum(((ydata - fitfunc(xdata, *para)) / dydata)**2)
	chi2red = chi2 / (len(ydata)-len(para))


	### plot distribution with fit
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	bar = plt.bar(bins, counts, width=binning, color=color_name, label='%s %s @ %s'%(name,beam_energies[i],thickness_titles[i]))
	fit_info = (r'total counts = %1d'%(total_counts) + '\n' +
        	    r'$\mu = %.3f\pm%.3f$'%(para[0],perr[0]) + '\n' +
            	    r'$\sigma = %.3f\pm%.3f$'%(abs(para[1]),perr[1])
		    )
	plt.text(0.15, 0.94, fit_info, fontsize=8, transform = ax.transAxes,
            	verticalalignment='top', horizontalalignment='left')
	x_fit = xdata
	y_fit = fitfunc(x_fit, *para)
	fit = plt.plot(x_fit, y_fit, ls='--', lw=1, alpha=0.8, color='red')
	plt.yscale('log')
	plt.ylim(1., 1e4)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'counts [#]')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/spectrum_%s_%s_%s_fit.png'%(name,beam_energies[i],thickness_names[i]))
	plt.close()
	with open('fitparameter_%s.txt'%name,'a') as outfile:
		outfile.write('%s,%s,%s,%.4f,%.4f,%.4f,%.4f\n'%(name,beam_energies[i],thickness_titles[i],para[0],perr[0],abs(para[1]),perr[1]))

	### plot normalized distribution with fit
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	counts_norm = counts/total_counts
	plt.bar(bins, counts_norm, width=binning, color=color_name, label='%s %s @ %s'%(name,beam_energies[i],thickness_titles[i]))
	fit_info = (r'$\mu = %.3f\pm%.3f$'%(para[0],perr[0]) + '\n' +
            	    r'$\sigma = %.3f\pm%.3f$'%(abs(para[1]),perr[1])
		    )
	plt.text(0.15, 0.94, fit_info, fontsize=8, transform = ax.transAxes,
            	verticalalignment='top', horizontalalignment='left')
	fit = plt.plot(x_fit, y_fit/total_counts, ls='--', lw=1, alpha=0.8, color='red')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(bin_low, bin_up)
	plt.savefig('./plots/spectrum_%s_%s_%s_normalized_fit.png'%(name,beam_energies[i],thickness_names[i]))
	plt.close()
