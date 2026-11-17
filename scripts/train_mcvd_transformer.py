"""Run the author's adapted 2000-epoch training path used by the paper."""
import argparse,json,sys,copy
import numpy as np
from pathlib import Path
from peas.common.paths import repository_root
root=repository_root();base=root/"baselines/mcvd_transformer"
sys.path.insert(0,str(base/"src"));sys.path.insert(0,str(base/"scripts"))
from mcvd_transformer.config import load_config
from mcvd_transformer.data import load_bundle
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene",required=True,choices=("canonical","eccentric","pipe"));p.add_argument("--seed",type=int);p.add_argument("--epochs",type=int,default=2000);p.add_argument("--device",default="cpu");p.add_argument("--output",type=Path)
a=p.parse_args()
with np.load(root/f"data/training/{a.scene}.npz") as z:profile=json.loads(str(z["metadata_json"]))["profile"]
c=load_config(root/f"configs/baselines/mcvd_transformer_{a.scene}.toml",profile=profile)
bundle=load_bundle(root/f"data/training/{a.scene}.npz",c)
meta=json.loads((root/f"configs/baselines/mcvd_transformer_{a.scene}_archive.json").read_text());seed=meta["seeds"][0] if a.seed is None else a.seed
out=a.output or root/f"outputs/mcvd_transformer/{a.scene}"
if a.scene=="canonical":
 import author_verified_canonical as trainer
 trainer.VERIFIED=out;c["seed"]=seed;c["training"]["epochs"]=a.epochs
 trainer.train_one(c,bundle,f"seed_{seed}",a.device)
else:
 import author_adapted_scenes as trainer
 trainer.OUT=out;trainer.train_one(c,bundle,seed,a.epochs,a.device)
