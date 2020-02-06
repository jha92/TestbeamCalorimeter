#! /usr/bin/env python
'''plot and analyse histograms from MSO scope

Usage:
    plot_spectrum_Tungsten.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

#energy calibration
p0 = -0.1705
p1 = 1.2392
calibfunc = lambda x: p0 + p1 * x

# INPUT
name = 'Tungsten'
beamenergy='2GeV'

#0.1mm
with open('../data/190308_170905.csv','r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_0pt1mm = calibfunc(bins)
max_0pt1mm = np.where(np.max(counts[5:])==counts)[0][0]
counts_0pt1mm = counts / np.sum(counts)

#1mm
with open('../data/190308_172713.csv','r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_1mm = calibfunc(bins)
max_1mm = np.where(np.max(counts[5:])==counts)[0][0]
counts_1mm = counts / np.sum(counts)

#5mm
with open('../data/190308_174439.csv','r') as infile:
    data = infile.readlines()[1:-1]
bins  = np.array([float(row.split(",")[0]) for row in data])
counts = np.array([float(row.split(",")[1]) for row in data])
bins = bins * -1.
binning = bins[1] - bins[0]
bins = bins - binning/2
counts = counts / 637
bins_5mm = calibfunc(bins)
max_5mm = np.where(np.max(counts[5:])==counts)[0][0]
counts_5mm = counts / np.sum(counts)

binning = bins_5mm[1] - bins_5mm[0]


fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.bar(bins_0pt1mm, counts_0pt1mm, width=binning, color='olive',       label='0.1mm', alpha=0.5)
plt.bar(bins_1mm,    counts_1mm,    width=binning, color='lawngreen',   label='1mm',   alpha=0.5)
plt.bar(bins_5mm,    counts_5mm,    width=binning, color='green',       label='5mm',   alpha=0.5)
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel(r'energy [GeV]')
plt.ylabel(r'normalized counts')
plt.xlim(0,3)
plt.legend(loc='upper right')
plt.savefig('./plots/spectrum_%s_%s_thicknessOverlay.png'%(name,beamenergy))
plt.close()

fig, ax = plt.subplots(figsize=(6,4), dpi=600)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.step(bins_0pt1mm, counts_0pt1mm, color='olive',     linewidth='1', label='0.1mm')
plt.step(bins_1mm   , counts_1mm,    color='lawngreen', linewidth='1', label='1mm')
plt.step(bins_5mm   , counts_5mm,    color='green',     linewidth='1', label='5mm')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel(r'energy [GeV]')
plt.ylabel(r'normalized counts')
plt.xlim(0,3)
plt.legend(loc='upper right')
plt.savefig('./plots/spectrum_%s_%s_thicknessOverlay2.png'%(name,beamenergy))
plt.close()





