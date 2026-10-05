#!/usr/bin/env python3
import csv, json, math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/scratch/kkumah/ERA5_temporal_matching_diagnostic_20260907')
OUT = ROOT / 'final'
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(parents=True, exist_ok=True)

raw = {}
summary = {}
for i in range(1, 5):
    td = ROOT / f'chunk_{i}' / 'tables'
    with open(td / 'raw_sufficient_statistics.json') as f:
        part = json.load(f)
    if set(raw) and set(part) != set(raw):
        raise RuntimeError(f'key mismatch in chunk {i}')
    for key, vals in part.items():
        dst = raw.setdefault(key, {k: 0 for k in vals})
        for name, value in vals.items(): dst[name] += value
    with open(td / 'change_summary.json') as f:
        s = json.load(f)
    for name in ('files','scans','shifted_scans','common_era5_footprints','changed_era5_footprints','absolute_change_sum','squared_change_sum'):
        summary[name] = summary.get(name, 0) + s[name]

summary['shifted_scan_percent'] = 100 * summary['shifted_scans'] / summary['scans']
summary['changed_value_percent'] = 100 * summary['changed_era5_footprints'] / summary['common_era5_footprints']
summary['mean_absolute_change_mm_h'] = summary['absolute_change_sum'] / summary['common_era5_footprints']
summary['rms_change_mm_h'] = math.sqrt(summary['squared_change_sum'] / summary['common_era5_footprints'])

def div(a,b): return a/b if b else float('nan')
rows=[]
for key, s in sorted(raw.items()):
    method, phase, reference, product = key.split('|')
    n=s['n']; denx=n*s['sum_x2']-s['sum_x']**2; deny=n*s['sum_y2']-s['sum_y']**2
    cc=(n*s['sum_xy']-s['sum_x']*s['sum_y'])/math.sqrt(denx*deny) if denx>0 and deny>0 else float('nan')
    rows.append(dict(method=method,phase=phase,reference=reference,product=product,n=n,
      POD=div(s['hits'],s['hits']+s['misses']),FAR=div(s['false_alarms'],s['hits']+s['false_alarms']),
      CSI=div(s['hits'],s['hits']+s['misses']+s['false_alarms']),frequency_bias=div(s['hits']+s['false_alarms'],s['hits']+s['misses']),
      CC=cc,RMSE=math.sqrt(div(s['sum_sqerr'],n)),relative_bias_percent=100*div(s['sum_x']-s['sum_y'],s['sum_y'])))

fields=list(rows[0])
with open(OUT/'tables'/'metrics_old_vs_corrected_2021.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(OUT/'tables'/'raw_sufficient_statistics_merged.json','w') as f: json.dump(raw,f,indent=2)
with open(OUT/'tables'/'change_summary_2021.json','w') as f: json.dump(summary,f,indent=2)

# Exact old-to-corrected deltas; GPROF rows provide a built-in invariance check.
idx={(r['method'],r['phase'],r['reference'],r['product']):r for r in rows}
metrics=['POD','FAR','CSI','frequency_bias','CC','RMSE','relative_bias_percent']
deltas=[]
for phase in ('rain','snow'):
  for ref in ('MRMS','StageIV'):
    for product in ('GPROF_V7','GPROF_V8','ERA5'):
      old=idx[('legacy',phase,ref,product)]; new=idx[('corrected',phase,ref,product)]
      deltas.append(dict(phase=phase,reference=ref,product=product,**{m:new[m]-old[m] for m in metrics}))
with open(OUT/'tables'/'metric_deltas_corrected_minus_legacy.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(deltas[0])); w.writeheader(); w.writerows(deltas)

era=[r for r in deltas if r['product']=='ERA5']
fig,axs=plt.subplots(2,4,figsize=(12,6.2)); axs=axs.ravel()
labels=[f"{r['phase'].title()}\n{r['reference']}" for r in era]
for ax,m in zip(axs,metrics):
    vals=[r[m] for r in era]; colors=['#1f5a85' if v>=0 else '#d17a22' for v in vals]
    ax.bar(labels,vals,color=colors,edgecolor='#333',linewidth=.6); ax.axhline(0,color='#555',lw=.8)
    ax.set_title(m.replace('_',' ')); ax.tick_params(axis='x',labelsize=8)
    for j,v in enumerate(vals): ax.text(j,v,f'{v:+.4f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=7)
axs[-1].axis('off')
fig.suptitle('ERA5 metric changes from corrected temporal matching (2021)',fontsize=13)
fig.text(.5,.01,'Corrected minus legacy; same GMI footprints, phase masks, thresholds, and references',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94)); fig.savefig(OUT/'figures'/'era5_metric_deltas_2021.png',dpi=200); plt.close(fig)

assert all(abs(r[m]) < 1e-12 for r in deltas if r['product']!='ERA5' for m in metrics), 'GPROF invariance failed'
print(json.dumps({'summary':summary,'era5_deltas':era,'gprof_invariance':'PASS'},indent=2))
