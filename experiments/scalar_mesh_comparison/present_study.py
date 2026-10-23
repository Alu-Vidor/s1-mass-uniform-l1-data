"""S1 plotting definitions loaded by preparation/plot_figures.py.
No time integration, article-text writer or author-local environment guard."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
S1 = HERE.parents[1]
ROOT = HERE.parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sci(x):
    mantissa, exponent = f'{x:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 8.0, 'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'axes.linewidth': 0.65, 'ps.fonttype': 42, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7), sharey=True)
    styles = [('union', 'Union', '#0072B2', 'o'), ('uniform', 'Uniform', '#D55E00', 's'), ('graded', 'Standard graded', '#009E73', '^')]
    for ax, alpha in zip(axes, (0.3, 0.7)):
        for mesh, label, color, marker in styles:
            selected = [lookup[alpha, 1e-08, m, mesh] for m in (16, 32, 64, 128, 256)]
            ax.loglog([r['N_actual'] for r in selected], [r['E_PL_sampled'] for r in selected], color=color, marker=marker, label=label)
        selected = [lookup[alpha, 1e-08, m, 'union'] for m in (64, 128, 256)]
        xx = np.array([r['N_actual'] for r in selected], float)
        guide = selected[0]['E_PL_sampled'] * 1.55 * (xx / xx[0]) ** (-(2 - alpha))
        ax.loglog(xx, guide, '--', color='0.35', linewidth=0.8)
        ax.text(xx[-1] * 0.81, guide[-1] * 1.35, f'$N^{{-{2 - alpha:.1f}}}$', fontsize=8.0, ha='center')
        ax.set_title(f'$\\alpha={alpha:.1f}$')
        ax.set_xlabel('Actual intervals $N$')
        ax.set_xlim(37, 950)
        ax.set_ylim(0.0001, 1.8)
        ax.set_xticks([50, 100, 200, 400, 800])
        ax.set_xticklabels(['50', '100', '200', '400', '800'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='major', linewidth=0.35, color='0.82')
        ax.tick_params(which='both', width=0.6)
    axes[0].set_ylabel('Sampled reconstruction error')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.005), handlelength=1.7, columnspacing=1.5)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.8, wspace=0.13)
if __name__ == '__main__':
    raise SystemExit('Run python preparation/plot_figures.py to generate figures from saved data.')
