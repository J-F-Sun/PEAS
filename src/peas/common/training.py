"""Unified orchestration; each scene retains its sampling and numerical objective."""
from __future__ import annotations
import copy
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import torch
from peas.common.config import save_config, validate_config
from peas.common.checkpoints import capture_rng, restore_rng, read_checkpoint, save_checkpoint, load_model
from peas.common.model import build_model
from peas.common.modules.steady import prepare_steady, retain_steady_asset
from peas.common.utils import set_seed, select_device, timestamp, weighted_total, save_json

def stages(c):
    tr=c['training']
    return [{**s,'loss_weights':s.get('loss_weights',tr.get('loss_weights',{}))}
            for s in (tr.get('lr_schedule') or [{'epochs':tr['epochs'],'lr':tr['lr']}])]

def clone_state(model):
    return {k:v.detach().clone() for k,v in model.transient.state_dict().items()}

def run_lbfgs(model,device,history,start_epoch):
    c=model.config
    tr=c['training']
    cfg=tr['lbfgs']
    batch=model.training.make_batch(device,sizes=cfg.get('batch_sizes',tr['batch_sizes']),
                                    adaptive_points=getattr(model,'adaptive_points',None))
    weights=cfg.get('loss_weights',tr.get('loss_weights',{}))
    val=tr.get('validation',{})
    val_batch=model.training.seeded_batch(device,val['batch_sizes'],val.get('seed',1073)) if val.get('enabled') else None
    best={'loss':float('inf'),'state':None}
    opt=torch.optim.LBFGS(model.transient.parameters(),lr=float(cfg['lr']),max_iter=int(cfg['max_iter']),
        max_eval=int(cfg['max_eval']),history_size=int(cfg['history_size']),
        tolerance_grad=float(cfg['tolerance_grad']),tolerance_change=float(cfg['tolerance_change']),
        line_search_fn=cfg['line_search_fn'])
    calls=0
    def validate():
        if val_batch is None: return
        parts=model.training.losses(model,val_batch)
        loss=float(weighted_total(parts,val.get('loss_weights',weights)).detach().cpu())
        if np.isfinite(loss) and loss<best['loss']:
            best['loss']=loss
            best['state']=clone_state(model)
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        parts=model.training.losses(model,batch)
        total=weighted_total(parts,weights)
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite L-BFGS loss')
        total.backward()
        calls+=1
        row={'epoch':start_epoch+calls,'phase':'lbfgs','total':float(total.detach().cpu()),
             **{k:float(v.detach().cpu()) for k,v in parts.items()}}
        history.append(row)
        if calls==1 or calls%int(cfg.get('log_every',25))==0:
            print(f"[{c['case_id']}] lbfgs_call={calls} total={row['total']:.6g}",flush=True)
        if val_batch is not None and (calls==1 or calls%int(val.get('log_every',10))==0): validate()
        return total
    opt.step(closure)
    validate()
    if val.get('restore_best_lbfgs') and best['state'] is not None:
        model.transient.load_state_dict(best['state'])
    return calls,None if best['state'] is None else best['loss']

def train(c, *, output_root=None, run_id=None, steady_checkpoint=None, init_checkpoint=None,
          resume=None, max_steps=None, smoke=False, skip_lbfgs=False):
    c=copy.deepcopy(c)
    validate_config(c)
    if max_steps is not None and max_steps<1: raise ValueError('max_steps must be positive')
    if smoke:
        tr=c['training']
        tr['epochs']=2
        tr['lr_schedule']=[{**stages(c)[0],'epochs':2}]
        tr['batch_sizes']={k:8 for k in tr['batch_sizes']}
        tr['adaptive_sampling']={'enabled':False}
        tr['lbfgs']={'enabled':False}
        tr['checkpoint_every']=1
    if init_checkpoint and resume: raise ValueError('Use either init_checkpoint or resume')
    device=select_device(c['runtime'])
    set_seed(int(c['seed']))
    history=[]
    epoch=0
    adaptive=None
    skipped=0
    opt=None
    active_stage=None
    restored=None
    if resume:
        model,payload=load_model(resume,device=device)
        stored=payload.get('training_state')
        if stored is None: raise ValueError('Old weights have no optimizer/RNG state; use --init-checkpoint')
        if stored['phase']=='complete': raise ValueError('This training run is already complete')
        # Config changes on resume would invalidate exact continuation.
        expected=copy.deepcopy(model.config)
        supplied=copy.deepcopy(c)
        for item in (expected,supplied):
            item.pop('paths',None)
            item['runtime'].pop('device',None)
            item['steady_state']['checkpoint']=None
        if expected!=supplied: raise ValueError('Resume config differs from saved training contract')
        c=model.config
        run=Path(resume).resolve().parent
        epoch=stored['epoch']
        history=stored['history']
        adaptive=stored['adaptive_points']
        if adaptive is not None: adaptive=tuple(p.to(device) for p in adaptive)
        skipped=stored.get('skipped_nonfinite',0)
        active_stage=stored.get('stage')
        restored=stored
        restore_rng(stored['rng'])
    else:
        rid=run_id or timestamp()
        if not rid or Path(rid).name!=rid or rid in ('.','..'):
            raise ValueError('run_id must be a single directory name')
        run=Path(output_root or c['paths']['output_root'])/rid
        run.mkdir(parents=True,exist_ok=False)
        if steady_checkpoint:
            c['steady_state']['checkpoint']=str(Path(steady_checkpoint).resolve())
        elif c['steady_state']['method']!='analytic' and not c['steady_state'].get('checkpoint'):
            c=prepare_steady(c,run/'steady_build')
        c=retain_steady_asset(c,run)
        model=build_model(c,device)
        if init_checkpoint:
            initial=read_checkpoint(init_checkpoint)
            old_model,_=load_model(init_checkpoint,config=c,steady_checkpoint=c['steady_state'].get('checkpoint'),device=device)
            model.transient.load_state_dict(old_model.transient.state_dict())
        save_config(c,run/'config.yaml')
        save_json(run/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
            'cuda':torch.version.cuda,'device':str(device),'seed':c['seed'],
            'initialization':'warm_start' if init_checkpoint else 'random',
            'feature_version':c['features']['version'],'smoke':smoke})
    model.transient.train()
    if restored:
        model.training.load_state_dict(restored.get('physics_training',{}))
    tr=c['training']
    schedule=stages(c)
    if restored and restored['optimizer'] is not None:
        opt=torch.optim.Adam(model.transient.parameters(),lr=schedule[active_stage]['lr'])
        opt.load_state_dict(restored['optimizer'])
        for state in opt.state.values():
            for key,val in state.items():
                if torch.is_tensor(val): state[key]=val.to(device)
    last_good=restored['last_good'] if restored else clone_state(model)
    if restored: last_good={k:v.to(device) for k,v in last_good.items()}
    adaptive_cfg=tr.get('adaptive_sampling',{})
    start=time.perf_counter()
    completed_this_call=0
    def save(phase,filename='last.pt',extra=None):
        state={'phase':phase,'epoch':epoch,'stage':active_stage,
               'optimizer':opt.state_dict() if opt else None,'history':history,
               'adaptive_points':adaptive,'skipped_nonfinite':skipped,'last_good':last_good,
               'physics_training':model.training.state_dict(),
               'rng':capture_rng(),**(extra or {})}
        save_checkpoint(run/filename,model,state)
        save_json(run/'history.json',history)
    def recover():
        nonlocal opt,adaptive,skipped
        if c['scenario'] not in ('concentric','eccentric'):
            raise FloatingPointError('Nonfinite Adam loss, gradient, or update')
        skipped+=1
        if skipped>tr.get('max_nonfinite_skips',50): raise FloatingPointError('Too many nonfinite steps')
        model.transient.load_state_dict(last_good)
        lr=max(opt.param_groups[0]['lr']*tr.get('nonfinite_lr_decay',0.5),tr.get('min_lr',1e-7))
        opt=torch.optim.Adam(model.transient.parameters(),lr=lr)
        adaptive=None
    offset=0
    for i,stage in enumerate(schedule):
        end=offset+int(stage['epochs'])
        if epoch>=end:
            offset=end
            continue
        if active_stage!=i or opt is None:
            if opt is None or tr.get('adam_reset_each_stage'):
                opt=torch.optim.Adam(model.transient.parameters(),lr=float(stage['lr']))
            else:
                for group in opt.param_groups: group['lr']=float(stage['lr'])
            active_stage=i
        while epoch<end:
            epoch+=1
            model.training.set_epoch(epoch-1)
            astart=int(adaptive_cfg.get('start_epoch',800))
            if adaptive_cfg.get('enabled') and epoch>=astart and (adaptive is None or (epoch-astart)%int(adaptive_cfg['refresh_every'])==0):
                adaptive=model.training.refresh_adaptive(model,device)
            opt.zero_grad(set_to_none=True)
            parts=model.training.losses(model,model.training.make_batch(device,adaptive_points=adaptive))
            total=weighted_total(parts,stage['loss_weights'])
            success=bool(torch.isfinite(total))
            if success:
                total.backward()
                clip=float(tr.get('grad_clip_norm',0))
                if clip>0: torch.nn.utils.clip_grad_norm_(model.transient.parameters(),clip)
                success=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.transient.parameters())
            if success:
                opt.step()
                success=all(bool(torch.isfinite(p).all()) for p in model.transient.parameters())
            if not success:
                recover()
                row={'epoch':epoch,'phase':'adam','skipped_nonfinite':True}
            else:
                if c['scenario'] in ('concentric','eccentric'): last_good=clone_state(model)
                row={'epoch':epoch,'phase':'adam','stage':i,'lr':opt.param_groups[0]['lr'],
                     'total':float(total.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in parts.items()}}
            history.append(row)
            completed_this_call+=1
            if epoch==1 or epoch==tr['epochs'] or epoch%int(tr.get('log_every',100))==0:
                print(f"[{c['case_id']}] adam={epoch}/{tr['epochs']} total={row.get('total')} elapsed={time.perf_counter()-start:.1f}s",flush=True)
            pending='lbfgs_pending' if epoch==tr['epochs'] and tr['lbfgs'].get('enabled') and not skip_lbfgs else 'adam'
            if epoch%int(tr.get('checkpoint_every',250))==0: save(pending)
            if max_steps is not None and completed_this_call>=max_steps and (epoch<tr['epochs'] or pending=='lbfgs_pending'):
                save(pending)
                return {'checkpoint':str(run/'last.pt'),'phase':pending,'epoch':epoch,'run_dir':str(run)}
        offset=end
    calls=0
    best=None
    if tr['lbfgs'].get('enabled') and not skip_lbfgs:
        # Resume is supported at this boundary; interrupted L-BFGS restarts its fixed batch from this RNG.
        model.adaptive_points=adaptive
        save('lbfgs_pending')
        calls,best=run_lbfgs(model,device,history,epoch)
    final_state={'lbfgs_calls':calls,'best_validation_loss':best}
    save('complete','last.pt',final_state)
    save('complete','model.pt',final_state)
    summary={'checkpoint':str(run/'model.pt'),'phase':'complete','epoch':epoch,'run_dir':str(run),
             'lbfgs_calls':calls,'best_validation_loss':best,'elapsed_seconds':time.perf_counter()-start,
             'fem_locator_fallback_points':getattr(model.steady.provider,'fallback_points',0)}
    save_json(run/'summary.json',summary)
    print(f"[{c['case_id']}] saved {run/'model.pt'}",flush=True)
    return summary
