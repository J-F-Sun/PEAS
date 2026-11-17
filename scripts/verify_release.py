"""Verify manifest hashes, paper metrics and every supplied inference checkpoint."""
import argparse,csv,hashlib,json,platform,importlib.metadata,tomllib
from pathlib import Path
import numpy as np
from peas.common.paths import repository_root
from peas.evaluation.archive import read,errors,SCENES,METHODS,VARIANTS

def verify(inference=False):
 root=repository_root();manifest=json.loads((root/'manifest.json').read_text());count=0
 for item in manifest['files']:
  path=root/item['path']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'Checksum mismatch: {path}')
  count+=1
 if platform.python_version()!=manifest['python']:raise ValueError('Use the Python version pinned in .python-version')
 with (root/'pyproject.toml').open('rb') as f:project=tomllib.load(f)
 for spec in project['project']['dependencies']:
  name,version=spec.split('==');actual=importlib.metadata.version(name)
  if actual!=version:raise ValueError(f'Dependency mismatch: {name}={actual}; expected {version}')
 expected=list(csv.DictReader((root/'results/expected_metrics.csv').open()))
 observed={}
 for group,methods in [('comparison',METHODS),('ablations',VARIANTS)]:
  for scene in SCENES:
   for method in methods:
    d=read(scene,method,group);e=errors(d['prediction'],d['reference']);means=e.mean(0)
    for ident,value in zip(d['ids'],means):observed[group,scene,method,str(ident)]=value
    observed[group,scene,method,'__mean__']=e.mean()
 for r in expected:np.testing.assert_allclose(observed[r['group'],r['scene'],r['method'],r['id']],float(r['mean_percent']),rtol=1e-11,atol=1e-11)
 # Compare against plotting audit values: retained original rows and updated checkpoint-audit rows.
 paper=list(csv.DictReader((root/'results/paper_plotted_metrics.csv').open()))
 for r in paper:np.testing.assert_allclose(observed['ablations' if r['figure']=='ablation' else r['figure'],r['scene'],r['method'],r['id']],float(r['relative_l2_percent']),rtol=1e-11,atol=1e-11)
 checks=[]
 if inference:
  import torch
  from peas.common.checkpoints import load_model
  torch.set_num_threads(1)
  for scene in SCENES:
   d=read(scene,'PEAS')
   for i,seed in enumerate(d['seeds']):
    model,_=load_model(root/f'checkpoints/peas/{scene}/seed_{seed}.pt',device='cpu')
    pred=model.predict(d['times'],d['samples']);delta=float(np.max(np.abs(pred-d['prediction'][i])))
    np.testing.assert_allclose(pred,d['prediction'][i],rtol=2e-4,atol=2e-6)
    checks.append(dict(scene=scene,seed=int(seed),max_abs_delta=delta))
 report={'hashes_verified':count,'metric_rows_verified':len(expected),'current_plot_points_verified':len(paper),'inference':checks,'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','matplotlib','PyYAML']}}
 out=root/'outputs/verification.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inference',action='store_true');a=p.parse_args();verify(a.inference)
