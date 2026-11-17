"""Apply one archived PEAS seed to the public test grid."""
import argparse
from pathlib import Path
import numpy as np,torch
from peas.common.paths import repository_root
from peas.common.checkpoints import load_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene",required=True,choices=("canonical","eccentric","pipe"));p.add_argument("--seed",type=int);p.add_argument("--device",default="cpu");p.add_argument("--output",type=Path)
a=p.parse_args();root=repository_root();torch.set_num_threads(1)
with np.load(root/f"data/predictions/comparison/{a.scene}/peas.npz") as z:seed=int(z["seeds"][0]) if a.seed is None else a.seed
model,_=load_model(root/f"checkpoints/peas/{a.scene}/seed_{seed}.pt",device=a.device)
with np.load(root/f"data/reference/{a.scene}.npz") as z:d={k:z[k] for k in z.files}
d["prediction"]=model.predict(d["times"],d["samples"]);d["seeds"]=np.array([seed]);out=a.output or root/f"outputs/predictions/{a.scene}_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**d);print(out)
