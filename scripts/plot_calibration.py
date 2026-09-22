"""Reliability diagram from measured SST-2 pilot bins; matplotlib required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
runs = [json.loads(p.read_text()) for p in (root/'results/pilot').glob('sst2__Qwen*/run.json')]
runs = [r for r in runs if r['status']=='complete' and r['config']['model']=='Qwen/Qwen2.5-0.5B-Instruct']
style = {'zero_shot':('Zero-shot','#778899'), 'few_shot':('Few-shot: 4/class','#2874a6'), 'lora':('LoRA: 4/class','#15927f')}
fig,axes=plt.subplots(1,2,figsize=(10.8,4.8))
axes[0].plot([0,1],[0,1],color='#aaaaaa',linestyle='--',linewidth=1,label='Perfect calibration')
for r in sorted(runs,key=lambda x:x['method']):
    m=r['metrics']; label,color=style[r['method']]
    bins=[b for b in m['reliability_bins'] if b['count']]
    axes[0].plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',markersize=4,color=color,label=label)
    axes[1].plot([b['mean_confidence'] for b in bins],[b['count']/m['n_probability_rows'] for b in bins],marker='o',markersize=4,color=color,label=label)
axes[0].set(xlabel='Mean top-label probability',ylabel='Empirical accuracy',xlim=(.48,1.01),ylim=(0,1.03),title='Reliability (15 equal-width bins)')
axes[1].set(xlabel='Mean top-label probability',ylabel='Fraction of evaluated examples',xlim=(.48,1.01),ylim=(0,1.03),title='Where the predictions fall')
for axis in axes:
    axis.grid(alpha=.15)
    axis.spines[['top','right']].set_visible(False)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.045),fontsize=9)
fig.suptitle('SST-2 pilot · Qwen 0.5B · 200 held-out examples',fontsize=14,x=.06,ha='left')
fig.text(.06,.005,'Restricted-label likelihood probabilities. Bin estimates can be noisy; no post-hoc calibration or test-tuned thresholds.',fontsize=8,color='#444444')
fig.tight_layout(rect=[0,.14,1,.92])
output=root/'results/figures'
output.mkdir(exist_ok=True)
fig.savefig(output/'sst2-reliability.png',dpi=180,bbox_inches='tight')
fig.savefig(output/'sst2-reliability.svg',bbox_inches='tight')
print(output/'sst2-reliability.png')
