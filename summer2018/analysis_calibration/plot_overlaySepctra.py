#! /usr/bin/env python
'''plot and analyse histograms from MSO scope

Usage:
    plot_overlaySpectra.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
name = 'Air'
color_name = 'deepskyblue'

data_1GeV = '../data/air/1_0Gev_180817.csv'
with open(data_1GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 500
bins_1GeV = bins
max_1GeV = np.where(np.max(counts)==counts)[0][0]
counts_1GeV = counts / np.sum(counts)

data_2GeV = '../data/air/2_0Gev_180817.csv'
with open(data_2GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 500
bins_2GeV = bins
max_2GeV = np.where(np.max(counts)==counts)[0][0]
counts_2GeV = counts / np.sum(counts)

data_3GeV = '../data/air/3_0Gev_180817.csv'
with open(data_3GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 500
bins_3GeV = bins
max_3GeV = np.where(np.max(counts)==counts)[0][0]
counts_3GeV = counts / np.sum(counts)

data_4GeV = '../data/air/4_0Gev_180817.csv'
with open(data_4GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 500
bins_4GeV = bins
max_4GeV = np.where(np.max(counts)==counts)[0][0]
counts_4GeV = counts / np.sum(counts)



### plot normalized distribution
fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
borderCut = 10
plt.bar(bins_1GeV[max_1GeV-borderCut:max_1GeV+borderCut], counts_1GeV[max_1GeV-borderCut:max_1GeV+borderCut], width=binning, color='#76EEC6', label='1GeV')
plt.bar(bins_2GeV[max_2GeV-borderCut:max_2GeV+borderCut], counts_2GeV[max_2GeV-borderCut:max_2GeV+borderCut], width=binning, color='#00B2EE', label='2GeV')
plt.bar(bins_3GeV[max_3GeV-borderCut:max_3GeV+borderCut], counts_3GeV[max_3GeV-borderCut:max_3GeV+borderCut], width=binning, color='#1874CD', label='3GeV')
plt.bar(bins_4GeV[max_4GeV-borderCut:max_4GeV+borderCut], counts_4GeV[max_4GeV-borderCut:max_4GeV+borderCut], width=binning, color='#104E8B', label='4GeV')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'normalized counts')
plt.xlim(0,4.5)
plt.legend()
plt.savefig('./plots/calibration_pulses_overlay.png')
plt.show()


