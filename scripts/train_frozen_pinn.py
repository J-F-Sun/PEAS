"""Fit the archived Frozen-PINN configuration, one source and seed at a time.
Run with baselines/frozen_pinn/.venv/bin/python; its NumPy 1.x environment is isolated.
"""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'baselines/frozen_pinn/src'))
os.environ.setdefault("MPLCONFIGDIR",str(root/"outputs/.mpl-frozen"))
from threadpoolctl import threadpool_limits
from frozen_pinn.common.solver import fit
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True,choices=('canonical','eccentric','pipe'));p.add_argument('--seed',type=int);p.add_argument('--source-id');p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path)
a=p.parse_args();archive=json.loads((root/'configs/baselines/frozen_pinn_archive.json').read_text());c=archive['configs'][a.scene];c['seed']=archive['seeds'][0] if a.seed is None else a.seed
with np.load(root/f'data/reference/{a.scene}_training_grid.npz') as z:times=z['times'].copy();ids=z['ids'].copy();samples=z['samples'].copy()
ident=a.source_id or str(ids[0]);index=list(ids).index(ident)
if a.smoke:
 c['model'].update(inner_basis=48,outer_basis=24);c['sampling'].update(interior_points=192,surface_points=48);c['evaluation'].update(surface_points=128,integration_points=51);c['time']['end']=.05;times=np.linspace(0,.05,5)
with threadpool_limits(limits=c['runtime']['threads']):
 model,diag=fit(c,samples[index]);pred,_=model.probability(times)
if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
out=a.output or root/f'outputs/frozen_pinn/{a.scene}/{ident}/seed_{c["seed"]}{"_smoke" if a.smoke else ""}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,prediction=pred[None,None],times=times,samples=samples[index:index+1],ids=ids[index:index+1],seeds=[c['seed']]);print(out)
