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

data_1GeV = '../data/190308_122427.csv'
with open(data_1GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_1GeV = bins
max_1GeV = np.where(np.max(counts)==counts)[0][0]
counts_1GeV = counts / np.sum(counts)

data_2GeV = '../data/190308_122947.csv'
with open(data_2GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_2GeV = bins
max_2GeV = np.where(np.max(counts)==counts)[0][0]
counts_2GeV = counts / np.sum(counts)

data_3GeV = '../data/190308_123553.csv'
with open(data_3GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_3GeV = bins
max_3GeV = np.where(np.max(counts)==counts)[0][0]
counts_3GeV = counts / np.sum(counts)

data_4GeV = '../data/190308_124247.csv'
with open(data_4GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_4GeV = bins
max_4GeV = np.where(np.max(counts)==counts)[0][0]
counts_4GeV = counts / np.sum(counts)

data_5GeV = '../data/190308_125516.csv'
with open(data_5GeV,'r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_5GeV = bins
max_5GeV = np.where(np.max(counts)==counts)[0][0]
counts_5GeV = counts / np.sum(counts)



### plot normalized distribution
fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
border_left = 30
border_right = 40
plt.bar(bins_1GeV[max_1GeV-border_left:max_1GeV+border_right], counts_1GeV[max_1GeV-border_left:max_1GeV+border_right], width=binning, color='#76EEC6', label='1GeV')
plt.bar(bins_2GeV[max_2GeV-border_left:max_2GeV+border_right], counts_2GeV[max_2GeV-border_left:max_2GeV+border_right], width=binning, color='#00B2EE', label='2GeV')
plt.bar(bins_3GeV[max_3GeV-border_left:max_3GeV+border_right], counts_3GeV[max_3GeV-border_left:max_3GeV+border_right], width=binning, color='#009ACD', label='3GeV')
plt.bar(bins_4GeV[max_4GeV-border_left:max_4GeV+border_right], counts_4GeV[max_4GeV-border_left:max_4GeV+border_right], width=binning, color='#1874CD', label='4GeV')
plt.bar(bins_5GeV[max_5GeV-border_left:max_5GeV+border_right], counts_5GeV[max_5GeV-border_left:max_5GeV+border_right], width=binning, color='#104E8B', label='5GeV')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'normalized counts')
plt.xlim(0,5.5)
plt.legend(loc='upper right')
plt.savefig('./plots/calibration_pulses_overlay.png')
plt.show()


