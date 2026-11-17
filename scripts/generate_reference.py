"""Generate an analytic U reference or a transient FEM S/C reference."""
import argparse,json
from pathlib import Path
import numpy as np
from peas.common.paths import repository_root
from peas.common.references import canonical_analytic,solve_fem,FEMSettings
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene",required=True,choices=("canonical","eccentric","pipe"));p.add_argument("--output",type=Path);p.add_argument("--dt",type=float,default=None);p.add_argument("--smoke",action="store_true")
a=p.parse_args();root=repository_root();c=json.loads((root/f"configs/peas/{a.scene}.json").read_text());g={**c["problem"]["domain"],**c["problem"]["boundary"]};g.setdefault("rx_center",[0,0,0]);g.setdefault("rx_radius",1.);g.setdefault("wall_kappa",g.get("kappa",0.))
with np.load(root/f"data/reference/{a.scene}.npz") as z:times=z["times"].copy();samples=z["samples"].copy();ids=z["ids"].copy()
if a.smoke:times=times[:3];samples=samples[:2];ids=ids[:2]
if a.scene=="canonical":q=canonical_analytic(samples,times,g);info={"method":"analytic erfc"}
elif a.scene=="eccentric":
 from peas.scenarios.eccentric.reference import solve
 # Historical paper reference: 3-D tetrahedral P1, lumped mass, implicit Euler.
 q,info=solve({"kind":"shell",**g},samples,times,mesh_size=.4 if not a.smoke else 1.5,dt=a.dt or .005)
 q=q.T
else:q,info=solve_fem(a.scene,samples,times,g,FEMSettings(dt=a.dt or .01),pipe_mesh=root/"data/steady/pipe.npz")
out=a.output or root/f"outputs/reference/{a.scene}{'_smoke' if a.smoke else ''}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,times=times,samples=samples,ids=ids,reference=q,metadata_json=json.dumps(info));print(out)
