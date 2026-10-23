"""S1 plotting definitions loaded by preparation/plot_figures.py.
No time integration, article-text writer or author-local environment guard."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
HERE = Path(__file__).resolve().parent
S1 = HERE.parents[1]

def sci(x):
    mantissa, exponent = f'{x:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'

def inline(x):
    return sci(x).replace('$', '\\(', 1)[:-1] + '\\)'

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 7.5, 'lines.linewidth': 1.1, 'lines.markersize': 3.3, 'axes.linewidth': 0.65, 'ps.fonttype': 42, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
    for alpha, color in ((0.3, '#0072B2'), (0.7, '#D55E00')):
        for delta, marker, style in (('1', 'o', '-'), ('1/100', 's', '--')):
            chosen = [lookup[alpha, delta, m] for m in (16, 32, 64, 128, 256)]
            axes[0].loglog([r['N'] for r in chosen], [r['nodal_error'] for r in chosen], marker=marker, linestyle=style, color=color, label=f'$\\alpha={alpha},\\ \\delta=' + ('1' if delta == '1' else '10^{-2}') + '$')
    for alpha, color, factor in ((0.3, '#0072B2', 0.62), (0.7, '#D55E00', 1.55)):
        chosen = [lookup[alpha, '1', m] for m in (64, 128, 256)]
        xx = np.array([r['N'] for r in chosen])
        yy = chosen[0]['nodal_error'] * factor * (xx / xx[0]) ** (-(2 - alpha))
        axes[0].loglog(xx, yy, ':', color=color, linewidth=0.8)
    axes[0].set_xlabel('Actual L1 intervals $N$')
    axes[0].set_ylabel('Maximum nodal error')
    axes[0].set_title('Independent-reference convergence')
    axes[0].legend(frameon=False, loc='lower left', handlelength=1.4, labelspacing=0.25)
    axes[0].set_xlim(48, 1550)
    ref = np.load(HERE / 'reference_a3_delta_1_100_K32768.npz')
    coarse = np.load(HERE / 'a3_delta_1_100_m32.npz')
    for i, color in enumerate(('#0072B2', '#D55E00', '#009E73')):
        selected = ref['nodes'] >= 1e-16
        axes[1].semilogx(ref['nodes'][selected], ref['U'][selected, i], color=color, label=f'$u_{i + 1}$')
        selected = np.flatnonzero(coarse['times'] >= 1e-16)[::5]
        axes[1].semilogx(coarse['times'][selected], coarse['U'][selected, i], linestyle='none', marker='o', markersize=2.6, markerfacecolor='white', markeredgewidth=0.6, color=color)
    axes[1].set_xlim(1e-16, 1)
    axes[1].set_xlabel('Time $t$')
    axes[1].set_ylabel('State')
    axes[1].set_title('$\\alpha=0.3,\\ \\delta=10^{-2}$')
    axes[1].legend(frameon=False, loc='upper right', handlelength=1.3)
    for ax in axes:
        ax.grid(True, which='major', linewidth=0.35, color='0.83')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='both', width=0.6)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.87, wspace=0.4)
if __name__ == '__main__':
    raise SystemExit('Run python preparation/plot_figures.py to generate figures from saved data.')
