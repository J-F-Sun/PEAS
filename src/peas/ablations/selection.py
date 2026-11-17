"""Validation selection for the two updated canonical backward ablations."""
import copy
from pathlib import Path

import numpy as np

from peas.common.checkpoints import capture_rng, restore_rng
from peas.common.paths import repository_root
from peas.common.utils import save_json
from peas.ablations import training


def train_selected(config, *, output_root, smoke=False):
    if config['scenario'] != 'canonical' or config['ablation']['variant'] not in ('no_features', 'no_steady'):
        raise ValueError('Validation selection applies to the updated canonical backward variants')
    with np.load(repository_root() / 'data/validation/canonical.npz', allow_pickle=False) as archive:
        validation = {k: archive[k].copy() for k in archive.files}
    candidates = {1, 2} if smoke else set(range(3500, 9001, 250))
    audit = {'scores': {}, 'best': None, 'best_epoch': None}
    original_save = training.save_checkpoint

    def selected_save(path, model, state, provenance):
        path = Path(path)
        epoch = state['epoch']
        if path.name == 'last.pt' and epoch in candidates and str(epoch) not in audit['scores']:
            rng = capture_rng()
            try:
                prediction = model.predict(validation['times'], validation['samples'])
                score = float((100 * np.linalg.norm(prediction-validation['reference'], axis=1)
                               / np.linalg.norm(validation['reference'], axis=1)).mean())
            finally:
                restore_rng(rng)
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite validation score')
            audit['scores'][str(epoch)] = score
            if audit['best'] is None or score < audit['best']:
                audit.update(best=score, best_epoch=epoch)
                original_save(path.parent / 'selected.pt', model,
                              {'phase': 'selected', 'epoch': epoch}, provenance)
            save_json(path.parent / 'validation.json', audit)
        original_save(path, model, dict(state, validation_selection=copy.deepcopy(audit)), provenance)

    training.save_checkpoint = selected_save
    try:
        return training.train(config, output_root=output_root, smoke=smoke)
    finally:
        training.save_checkpoint = original_save


def train_forward_selected(config, *, output_root, split, selection=None, smoke=False):
    """Select fixed-source U models on validation sources, then apply per-seed epochs."""
    if config['scenario']!='canonical' or config['ablation']['variant']!='no_reformulation':
        raise ValueError('Expected canonical forward configuration')
    with np.load(repository_root()/'data/validation/canonical.npz') as z:
        validation={k:z[k].copy() for k in z.files}
    candidates=[1,2] if smoke else list(range(3500,9001,250))
    chosen=None
    if split=='test':
        import json
        if selection is None:raise ValueError('Test-source training requires --selection from validation sources')
        chosen=int(json.loads(Path(selection).read_text())['epochs'][str(config['seed'])])
        if chosen not in candidates:raise ValueError('Selected epoch does not match this training budget')
        config=copy.deepcopy(config);config.setdefault('experiment',{})['selected_epoch']=chosen
    elif split!='validation':raise ValueError(split)
    audit={'scores':{},'best':None,'best_epoch':None}
    original_save=training.save_checkpoint
    def save(path,model,state,provenance):
        path=Path(path);epoch=state['epoch']
        if path.name=='last.pt' and epoch in candidates and str(epoch) not in audit['scores']:
            if chosen is not None:
                if epoch==chosen:original_save(path.parent/'selected.pt',model,dict(epoch=epoch,phase='selected'),provenance)
                audit['scores'][str(epoch)]=None
            else:
                ix=list(validation['ids']).index(config['ablation']['source_id']);rng=capture_rng()
                try:
                    pred=model.predict(validation['times'],validation['samples'][ix:ix+1])
                    score=float((100*np.linalg.norm(pred-validation['reference'][ix:ix+1],axis=1)/np.linalg.norm(validation['reference'][ix:ix+1],axis=1)).mean())
                finally:restore_rng(rng)
                if not np.isfinite(score):raise FloatingPointError('Nonfinite validation score')
                audit['scores'][str(epoch)]=score
                if audit['best'] is None or score<audit['best']:
                    audit.update(best=score,best_epoch=epoch)
                    original_save(path.parent/'selected.pt',model,dict(epoch=epoch,phase='selected'),provenance)
            save_json(path.parent/'validation.json',audit)
        original_save(path,model,dict(state,validation_selection=copy.deepcopy(audit)),provenance)
    training.save_checkpoint=save
    try:return training.train(config,output_root=Path(output_root)/split,smoke=smoke)
    finally:training.save_checkpoint=original_save


def select_forward_epochs(output_root, seeds, *, smoke=False):
    """Average all 20 validation sources; ties use the earliest candidate."""
    import json
    from peas.ablations.config import run_directory
    with np.load(repository_root()/'data/validation/canonical.npz') as z:ids=z['ids'].tolist()
    epochs=[1,2] if smoke else list(range(3500,9001,250));selected={}
    for seed in seeds:
        scores=[]
        for sid in ids:
            base=Path(output_root)/'validation'
            if smoke:base/='_smoke'
            run=base/'no_reformulation/canonical'/f'seed_{seed}'/sid
            if json.loads((run/'status.json').read_text())['phase']!='complete':raise ValueError(f'Incomplete validation source: {run}')
            summary=json.loads((run/'summary.json').read_text())
            if bool(summary['smoke'])!=smoke:raise ValueError('Smoke/formal records cannot be mixed')
            record=json.loads((run/'validation.json').read_text())['scores']
            if sorted(map(int,record))!=epochs:raise ValueError('Incomplete checkpoint candidates')
            scores.append([record[str(e)] for e in epochs])
        if not np.isfinite(scores).all():raise ValueError('Nonfinite validation scores')
        selected[str(seed)]=epochs[int(np.argmin(np.mean(scores,axis=0)))]
    result=dict(epochs=selected,candidates=epochs,validation_sources=len(ids),smoke=smoke)
    path=Path(output_root)/('selection_smoke.json' if smoke else 'selection.json');save_json(path,result)
    return result
