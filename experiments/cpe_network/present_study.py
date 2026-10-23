"""S1 plotting definitions loaded by preparation/plot_figures.py.
No time integration, article-text writer or author-local environment guard."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 8.0, 'axes.linewidth': 0.65, 'lines.linewidth': 1.1, 'lines.markersize': 3.6, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.4, 2.7))
    colors = ['#0072B2', '#D55E00', '#009E73']
    visible = times >= 1e-07
    for node, color in enumerate(colors):
        left.semilogx(times[visible], ideal_mV[visible, node], color=color, linewidth=1.15)
        left.semilogx(times[visible], finite_mV[visible, node], color=color, linestyle=(0, (3, 2)), linewidth=1.05)
        left.annotate(str(node + 1), (times[-1], ideal_mV[-1, node]), xytext=(3, 2), textcoords='offset points', color=color, fontsize=8.5, ha='left', va='center')
    left.set_xlim(1e-07, 0.067)
    left.set_ylim(-0.08, 7.15)
    left.set_xticks([1e-07, 1e-05, 0.001])
    left.set_yticks([0, 2, 4, 6])
    left.set_title('(a) Current-step response', pad=6)
    left.set_xlabel('Time (s)')
    left.set_ylabel('Node voltage (mV)')
    left.legend(handles=[Line2D([], [], color='0.2', linestyle='-', label='Ideal CPE'), Line2D([], [], color='0.2', linestyle=(0, (3, 2)), label='Finite RC')], loc='upper left', frameon=False, handlelength=1.8, borderaxespad=0.3)
    styles = [('uniform', 'Uniform', '#D55E00', 's', '-'), ('graded', 'Standard graded', '#009E73', '^', (0, (3, 2))), ('union', 'Union', '#0072B2', 'o', '-')]
    for mesh, label, color, marker, linestyle in styles:
        selected = [lookup[mesh, m] for m in levels]
        right.loglog([r['N'] for r in selected], [10000 * r['sampled_PL_error'] for r in selected], color=color, marker=marker, linestyle=linestyle, markerfacecolor='white', markeredgewidth=0.8, label=label)
    xx = np.array([550.0, 2300.0])
    guide = 60 * (xx / 550) ** (-1.5)
    right.loglog(xx, guide, linestyle='--', linewidth=0.8, color='0.4')
    right.text(1120, 32, '$N^{-1.5}$', fontsize=8.5, color='0.3')
    right.set_xlim(55, 3200)
    right.set_ylim(0.12, 2200)
    right.set_xticks([100, 1000])
    right.set_title('(b) Matched-budget L1 errors', pad=6)
    right.set_xlabel('Actual intervals $N$')
    right.set_ylabel('Sampled PL error ($\\mu$V)', labelpad=1)
    handles, labels = right.get_legend_handles_labels()
    order = [labels.index(x) for x in ['Union', 'Uniform', 'Standard graded']]
    right.legend([handles[k] for k in order], [labels[k] for k in order], loc='lower left', frameon=False, handlelength=1.65, borderaxespad=0.4, labelspacing=0.35, handletextpad=0.45)
    for ax in (left, right):
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='major', linewidth=0.35, color='0.85')
        ax.tick_params(which='both', width=0.6)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.2, top=0.86, wspace=0.44)
if __name__ == '__main__':
    raise SystemExit('Run python preparation/plot_figures.py to generate figures from saved data.')
