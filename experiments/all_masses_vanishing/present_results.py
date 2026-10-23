"""S1 plotting definitions loaded by preparation/plot_figures.py.
No time integration, article-text writer or author-local environment guard."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import PIL
HERE = Path(__file__).resolve().parent
S1 = HERE.parents[1]
ROOT = HERE.parents[1]
SOURCE = HERE / 'results.json'
TABLES = S1 / 'tables'
FIGURES = S1 / 'figures'
PREFIX = FIGURES / 'figure_3_all_masses_vanishing'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
MARKERS = ['o', 's', '^', 'D']
DELTAS = ['1', '1/100', '1/10000', '1/1000000']
DELTA_TEX = ['1', '10^{-2}', '10^{-4}', '10^{-6}']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sci(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False, 'svg.fonttype': 'none', 'svg.hashsalt': 's1-all-masses-frozen-v1', 'ps.fonttype': 42, 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'axes.linewidth': 0.65, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.2), sharey=True)
    fig.subplots_adjust(left=0.112, right=0.985, bottom=0.225, top=0.895, wspace=0.15)
    for ax, path, title in zip(axes, paths, ['(a) Fixed mass ratios', '(b) Changing mass ratios']):
        for delta, label, color, marker in zip(DELTAS, DELTA_TEX, COLORS, MARKERS):
            rows = grouped[path, delta]
            ax.loglog([r['actual_N'] for r in rows], [r['nodal_error'] for r in rows], color=color, marker=marker, markersize=3.8, linewidth=1.0, markerfacecolor='white', markeredgewidth=0.85, label=f'$\\delta={label}$')
        guide_n = np.array([100.0, 2000.0])
        ax.loglog(guide_n, 0.004 * (guide_n / 100.0) ** (-1.5), color='.25', linestyle=(0, (4, 2.5)), linewidth=0.9, label='$N^{-3/2}$ guide')
        ax.set_title(title, pad=6)
        ax.set_xlabel('Actual intervals $N$', labelpad=3)
        ax.set_xlim(25, 3500)
        ax.set_ylim(3e-05, 0.09)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='both', direction='in', top=True, right=True, width=0.6)
        ax.tick_params(which='major', length=3.2)
        ax.tick_params(which='minor', length=1.8)
        ax.grid(True, which='major', color='.85', linewidth=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Nodal error $E_N$', labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=5, columnspacing=0.85, handlelength=1.5, handletextpad=0.4, borderaxespad=0.0)
if __name__ == '__main__':
    raise SystemExit('Run python preparation/plot_figures.py to generate figures from saved data.')
