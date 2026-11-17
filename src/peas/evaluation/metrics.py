"""Run-level error summaries; equal-weight positions; paired MC bootstrap."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t
from .archive import SCENES,METHODS,VARIANTS,read,errors,positions

def summarize(bootstrap=2048):
    rows=[]
    for group,methods in (("comparison",METHODS),("ablations",VARIANTS)):
        for scene in SCENES:
            ids,dist,order=positions(scene)
            for method in methods:
                d=read(scene,method,group);e=errors(d["prediction"],d["reference"])[:,order]
                distribution=e;kind="Student t CI of mean across model runs"
                if "run_predictions" in d:
                    raw=d["run_predictions"][:,order];n=len(raw)
                    np.testing.assert_allclose(raw.mean(0),d["prediction"][0,order],rtol=1e-12,atol=1e-15)
                    if bootstrap<2: raise ValueError("MC bootstrap needs >=2 replicates")
                    rng=np.random.default_rng(20261114);draws=[]
                    for start in range(0,bootstrap,64):
                        weights=rng.multinomial(n,np.full(n,1/n),size=min(64,bootstrap-start))/n
                        mean=(weights@raw.reshape(n,-1)).reshape(-1,*raw.shape[1:])
                        draws.append(errors(mean,d["reference"][order]))
                    distribution=np.concatenate(draws);kind="Percentile bootstrap CI of error of 128-run mean"
                samples=np.column_stack([distribution,distribution.mean(1)]);mean=np.r_[e.mean(0),e.mean()];sd=samples.std(0,ddof=1)
                if "run_predictions" in d:lo,hi=np.quantile(samples,[.025,.975],axis=0)
                else:
                    half=t.ppf(.975,len(samples)-1)*sd/np.sqrt(len(samples));lo,hi=mean-half,mean+half
                for i,ident in enumerate([*ids,"__mean__"]):
                    rows.append(dict(group=group,scene=scene,method=method,id=str(ident),distance=float(dist[i]) if i<len(ids) else "",mean_percent=float(mean[i]),std_percent=float(sd[i]),ci95_low_percent=float(lo[i]),ci95_high_percent=float(hi[i]),n_runs=len(e),interval=kind))
    return rows

def write_metrics(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
