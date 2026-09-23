"""Render every eligible curve from the audited dashboard aggregate snapshot.

Requires matplotlib (optional, only for figures). No inference or paid calls.
"""
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'dashboard/dist/review-data.json'
    data=json.loads(source.read_text())
    conditions=[c for c in data['conditions'] if c['curve']]
    if not conditions:
        raise ValueError('No eligible complete conditions are available')
    conditions.sort(key=lambda c:(0 if 'Qwen3' in c['source_model'] and c['shots_per_class']==4 else 1,
                                   c['dataset'],c['source_model']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#bbc5d4','text.color':'#14223b',
        'axes.labelcolor':'#14223b','xtick.color':'#52627a','ytick.color':'#52627a',
        'svg.hashsalt':'jev-review-value-v1'})
    rows=math.ceil(len(conditions)/2)
    fig,axes=plt.subplots(rows,2,figsize=(12,4.15*rows+3.2),facecolor='white',squeeze=False)
    fig.subplots_adjust(left=.075,right=.975,top=.815,bottom=.185,wspace=.22,hspace=.42)
    fig.text(.075,.955,'When should an LLM decision be reviewed?',fontsize=24,weight='bold')
    fig.text(.075,.92,f'Jev review of numerical classification · all {len(conditions)} eligible conditions',fontsize=14,color='#52627a')
    fig.legend(handles=[Line2D([0],[0],color='#008d99',lw=2.5,marker='o',label='Least-confident rows first'),
                        Line2D([0],[0],color='#8570c4',lw=2,ls='--',label='Random selection expectation'),
                        Line2D([0],[0],color='#b27324',lw=1.8,ls=':',label='Jev alone')],
               loc='upper left',bbox_to_anchor=(.068,.90),ncol=3,frameon=False,fontsize=11)
    for ax,c in zip(axes.flat,conditions):
        p=c['curve']; x=[a['actual_review_fraction'] for a in p]
        ax.plot(x,[a['metrics']['micro_accuracy'] for a in p],color='#008d99',lw=2.5,marker='o',ms=5,zorder=3)
        ax.plot(x,[a['random_matched_rate']['micro_accuracy'] for a in p],color='#8570c4',ls='--',lw=2)
        ax.axhline(c['metrics']['direct_jev']['micro_accuracy'],color='#b27324',ls=':',lw=1.8)
        name={'Qwen/Qwen3-4B-Instruct-2507':'Qwen3 4B',
              'Qwen/Qwen2.5-0.5B-Instruct':'Qwen2.5 0.5B',
              'ibm-granite/granite-3.3-2b-instruct':'Granite 3.3 2B',
              'HuggingFaceTB/SmolLM2-1.7B-Instruct':'SmolLM2 1.7B'}[c['source_model']]
        dataset='Breast Cancer' if c['dataset']=='breast_cancer' else 'Wine'
        shots='4 examples per class' if c['shots_per_class'] else 'zero-shot'
        ax.set_title(f'{dataset} · {name}\n{shots} · n = {c["n_rows"]}',loc='left',fontsize=13,pad=14)
        ax.set(xlim=(-.02,1.03),ylim=(0,1.05),xlabel='Fraction of rows reviewed',ylabel='Accuracy')
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='y',color='#e6ebf2',zorder=0)
        half=p[3]; always=p[-1]
        text=f'At 50%: {half["metrics"]["micro_accuracy"]:.1%}\nFull review: {always["metrics"]["micro_accuracy"]:.1%}'
        ax.text(.96,.08,text,transform=ax.transAxes,ha='right',va='bottom',fontsize=11,color='#14223b',
                bbox={'facecolor':'#f3f5fa','edgecolor':'none','pad':6})
    for ax in list(axes.flat)[len(conditions):]:
        fig.delaxes(ax)
    fig.text(.075,.105,'Each panel reports the same six fixed coverage points and matched random reference.\nEligibility is descriptive; the separate sensitivity appendix includes constant-label sources.',fontsize=12,linespacing=1.5)
    fig.text(.075,.045,'Exploratory simulation from cached test outcomes; one split, no new calls. Source inference is still required for every row.\nNo validated gate, dollar savings or latency savings claim. Constant-label and incomplete conditions are excluded.\nData + method: github.com/statsguysam/jev-classification-benchmark · results/review_value/FINDINGS.md',fontsize=9,color='#52627a',linespacing=1.5)
    output=ROOT/'results/review_value/figures';output.mkdir(exist_ok=True)
    fig.savefig(output/'selective_review.png',dpi=180,facecolor='white',metadata={'Software':'matplotlib'})
    fig.savefig(output/'selective_review.svg',facecolor='white',metadata={'Date':None})
    metadata={'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'matplotlib_version':matplotlib.__version__, 'data_scope':f'All {len(conditions)} eligible conditions; accuracy only',
              'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix in {'.png','.svg'}}}
    (output/'MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output)

if __name__=='__main__':main()
