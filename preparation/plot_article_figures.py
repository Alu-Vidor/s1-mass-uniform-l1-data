"""S1 plotting definitions loaded by preparation/plot_figures.py.
No time integration, article-text writer or author-local environment guard."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent / 'figures'
TABLES = HERE.parent / 'tables'
COLORS = {'M1': '#4477AA', 'M2': '#EE7733', 'M3': '#228833', 'M4': '#AA3377', 'M5': '#66AADD'}
MARKERS = {'M1': 'o', 'M2': 's', 'M3': '^', 'M4': 'D', 'M5': 'v'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    with (TABLES / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def vals(rows, key):
    return [float(r[key]) for r in rows]

def style(ax):
    ax.tick_params(direction='in', top=True, right=True, length=3, width=0.6)
    ax.grid(True, which='major', color='.87', linewidth=0.45)
    ax.set_axisbelow(True)

def series(ax, rows, case, x, y):
    ax.plot(vals(rows, x), vals(rows, y), color=COLORS[case], marker=MARKERS[case], markersize=3.4, linewidth=1.0, markerfacecolor='white', markeredgewidth=0.75, label=case)

def guide(ax, x0, x1, y0):
    ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** (-1.5)], '--', color='.2', linewidth=0.9, label='$N^{-3/2}$ guide')

def save(fig, name, outputs):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * fig.dpi
    outside = []
    texts = []
    for ax in fig.axes:
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo <= tick.get_loc() <= hi:
                    texts.extend([tick.label1, tick.label2])
    for item in texts:
        if not item.get_visible() or not item.get_text():
            continue
        extent = item.get_window_extent(renderer)
        if extent.x0 < -1 or extent.y0 < -1 or extent.x1 > width + 1 or (extent.y1 > height + 1):
            outside.append(item.get_text())
    assert not outside, f'Clipped text in {name}: {outside}'
    for ext in ('png', 'eps'):
        path = MANUSCRIPT / f'{name}.{ext}'
        fig.savefig(path, dpi=1200)
        outputs[path.name] = sha(path)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'axes.linewidth': 0.65, 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
    outputs = {}
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.4, 3.2), layout='constrained')
    for case in COLORS:
        series(left, [r for r in linear if r['case'] == case], case, 'N_actual', 'E_node')
    left.set(xscale='log', yscale='log', xlabel='Actual intervals $N$', ylabel='Nodal error $E_m$', title='(a) Linear chain')
    guide(left, 100, 2600, 0.011)
    left.legend(loc='lower left', ncol=2, columnspacing=0.6, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    for case in ('M2', 'M3'):
        series(right, [r for r in linear if r['case'] == case], case, 'N_actual', 'E_N_to_1_5')
    right.set(xscale='log', xlabel='Actual intervals $N$', ylabel='Scaled error $E_mN^{3/2}$', title='(b) Coincident / nearby scales')
    right.legend(loc='upper left', handlelength=1.5)
    for ax in (left, right):
        style(ax)
    save(fig, 'Fig1', outputs)
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.4, 3.2), layout='constrained')
    for case in ('M2', 'M4', 'M5'):
        series(left, [r for r in nonlinear if r['case'] == case], case, 'coarse_N_actual', 'D_PL')
    left.set(xscale='log', yscale='log', xlabel='Coarse intervals $N_m$', ylabel='PL difference $D_m$', title='(a) Nonlinear refinement')
    guide(left, 70, 1100, 0.03)
    left.legend(loc='lower left', handlelength=1.5, labelspacing=0.3)
    n = vals(exact, 'N_actual')
    for key, marker, color, label in (('E_node', 'o', '#4477AA', '$E_{\\rm node}$'), ('E_PL', 's', '#EE7733', '$E_{\\rm PL}$'), ('total_budget', '^', '#228833', '$d_N+\\zeta_*$')):
        right.plot(n, vals(exact, key), marker=marker, markersize=3.4, linewidth=1.0, color=color, markerfacecolor='white', markeredgewidth=0.75, label=label)
    right.set(xscale='log', yscale='log', xlabel='Uniform intervals $N$', ylabel='Error / budget', title='(b) Manufactured solution')
    right.set_xticks(n, labels=[str(int(v)) for v in n])
    right.xaxis.set_minor_formatter(NullFormatter())
    guide(right, 25, 270, 0.0012)
    right.legend(loc='lower left', handlelength=1.5, labelspacing=0.3)
    for ax in (left, right):
        style(ax)
    save(fig, 'Fig2', outputs)
if __name__ == '__main__':
    raise SystemExit('Run python preparation/plot_figures.py to generate figures from saved data.')
