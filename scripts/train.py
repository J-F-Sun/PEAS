"""Train PEAS or an ablation under the current plotted protocol. Default is one run."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from peas.common.paths import repository_root
from peas.common.config import load_config

def main():
    root=repository_root();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",choices=("canonical","eccentric","pipe"),required=True)
    p.add_argument("--variant",choices=("peas","no_reformulation","no_steady","no_features"),default="peas")
    p.add_argument("--seed",type=int);p.add_argument("--source-id");p.add_argument("--device",default="cpu");p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,default=root/"outputs/training")
    p.add_argument("--split",choices=("validation","test"),default="test");p.add_argument("--selection",type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    if a.variant!="peas":
        from peas.ablations.config import make_config,formal_seeds
        from peas.ablations.training import train
        c=make_config(a.scene,a.variant,a.seed if a.seed is not None else formal_seeds(a.scene)[0],source_id=a.source_id,device=a.device,split=a.split)
        if a.scene=="canonical" and a.variant in ("no_features","no_steady"):
            from peas.ablations.selection import train_selected
            result=train_selected(c,output_root=a.output,smoke=a.smoke)
        elif a.scene=="canonical" and a.variant=="no_reformulation":
            from peas.ablations.selection import train_forward_selected
            result=train_forward_selected(c,output_root=a.output,split=a.split,selection=a.selection,smoke=a.smoke)
        else:
            result=train(c,output_root=a.output,smoke=a.smoke)
    else:
        import peas.common.training as training
        from peas.common.checkpoints import save_checkpoint
        from peas.evaluation.archive import errors
        c=load_config(root/f"configs/peas/{a.scene}.json");c["runtime"]["device"]=a.device
        if a.seed is not None:c["seed"]=a.seed
        run_id=f'{a.scene}_seed_{c["seed"]}'+("_smoke" if a.smoke else "")
        if a.scene=="canonical" and not a.smoke:
            with np.load(root/"data/validation/canonical.npz",allow_pickle=False) as z:validation={k:z[k].copy() for k in z.files}
            best=float("inf");seen=set();history=[];run=a.output.resolve()/run_id
            def selected_save(path,model,state):
                nonlocal best
                save_checkpoint(path,model,state);epoch=state["epoch"]
                if epoch>=3500 and epoch%250==0 and epoch not in seen:
                    seen.add(epoch);score=float(errors(model.predict(validation["times"],validation["samples"]),validation["reference"]).mean())
                    history.append({"epoch":epoch,"mean_percent":score})
                    if score<best:best=score;save_checkpoint(run/"best_validation.pt",model,state)
                    (run/"validation_history.json").write_text(json.dumps(history,indent=2))
            training.save_checkpoint=selected_save
        result=training.train(c,output_root=a.output,run_id=run_id,steady_checkpoint=c["steady_state"]["checkpoint"],smoke=a.smoke)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
