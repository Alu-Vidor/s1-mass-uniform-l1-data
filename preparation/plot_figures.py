"""Journal-only derivatives of frozen S1 figures, without solver execution.

Run with python from repository root.
The original presentation scripts are read and their plot statements alone
are compiled via AST; their table, manuscript and manifest writers never run.
Original experiment outputs and scientific input hashes are preserved.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
S1=HERE.parent
MANUSCRIPT=S1/'figures'
ROOT=S1
RECORDS=S1/'provenance/figure_generation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files():
    files=[p for p in (S1/'experiments').rglob('*') if p.is_file()
           and '__pycache__' not in p.parts and p.suffix.lower() in ('.npz','.json','.csv','.py','.tex','.texfrag','.log')]
    files += list((S1/'tables').glob('*.csv'))
    return {str(p.relative_to(S1)):sha(p) for p in sorted(files)}


def plot_statements(path, stop):
    """Select only main's plot statements; no old main or writer is called."""
    text=path.read_text(encoding='utf-8-sig')
    tree=ast.parse(text,filename=str(path))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(k for k,n in enumerate(main.body)
               if 'plt.rcParams.update' in ast.unparse(n))
    chosen=[]
    saves=0
    for n in main.body[start:]:
        code=ast.unparse(n)
        if stop=='two_save_calls':
            chosen.append(n)
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='save':
                saves+=1
                if saves==2:break
        elif ((stop=='canvas' and code.startswith('fig.canvas.draw')) or
              (stop=='targets' and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='targets' for t in n.targets)) or
              (stop=='for_ext' and isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ext') or
              (stop=='outputs' and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='outputs' for t in n.targets))):
            break
        else:chosen.append(n)
    if stop=='two_save_calls':assert saves==2
    assert chosen and not any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)
                              and c.func.attr in ('write_text','write_bytes','savefig','open')
                              for n in chosen for c in ast.walk(n))
    return compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec')


def source_namespace(path):
    # run_name is deliberately different from __main__: main cannot run.
    return runpy.run_path(str(path),run_name='journal_derivative_source')


def figures_from_sources():
    sources=[]
    path=HERE/'plot_article_figures.py';ns=source_namespace(path);sources.append(path)
    ns.update(dict(zip(('linear','nonlinear','exact'),
                       [ns['read'](name) for name in ('linear_mass3.csv','nonlinear_mass3.csv','manufactured_L1.csv')])))
    captured=[]
    ns.update(outputs={},save=lambda fig,name,outputs:captured.append((name,fig)))
    exec(plot_statements(path,'two_save_calls'),ns)
    figures=dict(captured)

    path=S1/'experiments/all_masses_vanishing/present_results.py';ns=source_namespace(path);sources.append(path)
    runs=json.loads(ns['SOURCE'].read_text(encoding='utf-8'))['runs']
    paths=['fixed_ratios','changing_ratios']
    ns.update(paths=paths,grouped={(p,d):sorted([r for r in runs if r['path']==p and r['delta_exact']==d],key=lambda r:r['m']) for p in paths for d in ns['DELTAS']})
    exec(plot_statements(path,'canvas'),ns);figures['Fig3']=ns['fig']

    path=S1/'experiments/scalar_mesh_comparison/present_study.py';ns=source_namespace(path);sources.append(path)
    rows=json.loads((path.parent/'results.json').read_text(encoding='utf-8'))
    ns['lookup']={(r['alpha'],r['mass'],r['m'],r['mesh']):r for r in rows}
    exec(plot_statements(path,'targets'),ns);figures['Fig4']=ns['fig']

    path=S1/'experiments/nonlinear_reference/present_study.py';ns=source_namespace(path);sources.append(path)
    rows=json.loads((path.parent/'results.json').read_text(encoding='utf-8'))['rows']
    ns['lookup']={(r['alpha'],r['delta'],r['m']):r for r in rows}
    exec(plot_statements(path,'for_ext'),ns);figures['Fig5']=ns['fig']

    path=S1/'experiments/cpe_network/present_study.py';ns=source_namespace(path);sources.append(path)
    report=json.loads((path.parent/'results.json').read_text(encoding='utf-8'))
    with np.load(path.parent/'physical_model_comparison.npz',allow_pickle=False) as z:
        ns.update(times=z['times_s'].copy(),ideal_mV=1000*z['ideal_CPE_volts'].copy(),
                  finite_mV=1000*z['finite_RC_volts'].copy())
    ns.update(report=report,runs=report['runs'],levels=sorted({r['m'] for r in report['runs']}),
              lookup={(r['mesh'],r['m']):r for r in report['runs']})
    exec(plot_statements(path,'outputs'),ns);figures['Fig6']=ns['fig']
    return figures,sources


def coordinates_signature(fig):
    values=[]
    for ax in fig.axes:
        values.append((ax.get_xlim(),ax.get_ylim(),
            [(np.asarray(line.get_xdata()).tolist(),np.asarray(line.get_ydata()).tolist()) for line in ax.lines]))
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def journal_style(name,fig):
    before=coordinates_signature(fig)
    old_titles=[]
    for i,ax in enumerate(fig.axes):
        old_titles.append(ax.get_title())
        for loc in ('left','center','right'):ax.set_title('',loc=loc)
        ax.set_title(f'({chr(97+i)})',loc='left',fontsize=9.,pad=5)
    # Distinguishable component curves/markers for the new nonlinear study.
    # Only visual styles change: numerical coordinates are checked below.
    if name=='Fig5':
        fig.subplots_adjust(right=.965,top=.76)
        ax=fig.axes[0]
        data=[line for line in ax.lines if line.get_label().startswith('$\\alpha=')]
        assert len(data)==4
        for line,marker,style in zip(data,('o','s','^','D'),('-','--','-.',':')):
            line.set_marker(marker);line.set_linestyle(style)
        handles,labels=ax.get_legend_handles_labels()
        ax.get_legend().remove()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.015),
                   ncol=2,frameon=False,handlelength=1.6,columnspacing=1.5,fontsize=8.)
        ax=fig.axes[1]
        for node,(style,marker) in enumerate(zip(('-','--','-.'),('o','s','^'))):
            ax.lines[2*node].set_linestyle(style)
            ax.lines[2*node+1].set_marker(marker)
        ax.legend(frameon=False,loc='upper right',handlelength=1.3,fontsize=8.)
    for text in fig.findobj(match=Text):
        if text.get_text():
            text.set_fontsize(min(10.,max(8.,text.get_fontsize())))
            # All figure lettering is black (contrast21:1 on white).
            text.set_color('black')
    assert coordinates_signature(fig)==before,'Numerical coordinates changed while styling'
    fig.canvas.draw()
    return old_titles,before


def visible_texts(fig):
    for ax in fig.axes:
        yield ax.xaxis.label;yield ax.yaxis.label
        for title in (ax.title,ax._left_title,ax._right_title):yield title
        for text in ax.texts:yield text
        legend=ax.get_legend()
        if legend:
            yield from legend.get_texts()
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            lo,hi=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if lo<=tick.get_loc()<=hi:
                    if tick.label1.get_visible():yield tick.label1
                    if tick.label2.get_visible():yield tick.label2
    for legend in fig.legends:yield from legend.get_texts()


def export(name,fig,old_titles,signature):
    renderer=fig.canvas.get_renderer();bbox=fig.bbox
    outside=[];sizes=[]
    for text in visible_texts(fig):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer);sizes.append(text.get_fontsize())
            if b.x0<bbox.x0-1 or b.y0<bbox.y0-1 or b.x1>bbox.x1+1 or b.y1>bbox.y1+1:
                outside.append(text.get_text())
    assert not outside,f'{name} clipped lettering: {outside}'
    png=MANUSCRIPT/f'{name}.png';eps=MANUSCRIPT/f'{name}.eps'
    fig.savefig(png,dpi=1200,facecolor='white')
    with Image.open(png) as im:
        rgb=im.convert('RGB');rgb.save(png,dpi=(1200,1200))
    fig.savefig(eps,dpi=1200,facecolor='white')
    with Image.open(png) as im:
        mode=im.mode;pixels=list(im.size);dpi=list(im.info['dpi'])
        assert mode=='RGB' and pixels[0]>=1500 and pixels[1]>=1200
        preview=im.copy();preview.thumbnail((1620,960),Image.Resampling.LANCZOS)
        preview.save(RECORDS/f'figure_{name[3:]}_review.png')
    text=eps.read_text(encoding='ascii',errors='ignore')
    assert '/FontType 42 def' in text and '/FontType 3 def' not in text
    assert min(sizes)>=8 and max(sizes)<=12
    record={'removed_internal_titles':old_titles,'retained_panel_tags':['(a)','(b)'],
        'png_pixels':pixels,'png_mode':mode,'png_bits_per_channel':8,'png_dpi':dpi,
        'physical_inches':list(fig.get_size_inches()),
        'lettering_points':[min(sizes),max(sizes)],'lettering_contrast_on_white':21,
        'eps_embedded_font_types':[42],'text_clipping_check':'passed',
        'numerical_line_coordinate_sha256':signature,
        'numerical_line_coordinates_unchanged':True,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (png,eps)}}
    plt.close(fig)
    return record


def main():
    # Use any environment satisfying requirements-plot.txt.
    RECORDS.mkdir(parents=True,exist_ok=True)
    MANUSCRIPT.mkdir(parents=True,exist_ok=True)
    protected=protected_files()
    figures,sources=figures_from_sources()
    figures.pop('Fig2', None)  # Historical diagnostic is not used in the final article.
    assert sorted(figures)==['Fig1','Fig3','Fig4','Fig5','Fig6']
    records={}
    for name,fig in figures.items():
        titles,signature=journal_style(name,fig)
        records[name]=export(name,fig,titles,signature)
    after=protected_files()
    assert protected==after,'Frozen experimental files changed during derivative generation'
    provenance={'status':'PASS','purpose':'Submission-format derivatives only',
        'source_guidelines':'https://link.springer.com/journal/11075/submission-guidelines',
        'script_sha256':sha(__file__),'original_presenter_sha256':{str(p.relative_to(S1)):sha(p) for p in sources},
        'protected_scientific_file_count':len(protected),'protected_scientific_sha256':protected,
        'all_protected_scientific_files_unchanged':True,
        'plot_statements_only':True,'original_main_functions_called':False,
        'solver_calls':0,'original_presenters_manifests_and_tables_modified':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
            'presentation_runtime_separate_from_scientific_runtime':True},
        'figures':records}
    table_path=RECORDS/'figure_dimensions.csv'
    with table_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['figure','pixels_x','pixels_y','mode','bits_per_channel','dpi','width_inches','height_inches','minimum_font_pt','maximum_font_pt','EPS_font_type'])
        for name,r in records.items():
            writer.writerow([name,*r['png_pixels'],r['png_mode'],r['png_bits_per_channel'],1200,*r['physical_inches'],*r['lettering_points'],42])
    provenance['dimension_table_sha256']=sha(table_path)
    (RECORDS/'figure_compliance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','figures':list(records),'protected_files':len(protected)},indent=2))


if __name__=='__main__':main()
