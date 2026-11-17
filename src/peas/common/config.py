"""Versioned independent scene configurations and cwd-independent paths."""
from __future__ import annotations
import copy
import json
import os
import re
from pathlib import Path
import numpy as np
import yaml
from peas.common.modules.features import FEATURES, validate_sphere_config
from peas.common.paths import repository_root

SCENARIOS = ('canonical', 'concentric', 'eccentric', 'pipe')
PRESETS = {'canonical':'canonical3', 'concentric':'axis10_arrival',
           'eccentric':'axis10_arrival', 'pipe':'symmetric11_arrival'}

def validate_config(c):
    required = {'schema_version','scenario','case_id','problem','features','model',
                'steady_state','training','sampling','residual','evaluation','runtime','paths','seed'}
    if required - c.keys():
        raise ValueError(f'Missing configuration sections: {sorted(required-c.keys())}')
    if c['schema_version'] != 1 or c['scenario'] not in SCENARIOS:
        raise ValueError('Unsupported schema version or scenario')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', c['case_id']):
        raise ValueError('case_id must be a lowercase identifier')
    feat, model, scene = c['features'], c['model'], c['scenario']
    if 'representation' in c and (scene != 'concentric' or c['representation'] != 'radial'):
        raise ValueError('The radial variant requires the concentric scene')
    allowed_presets = ('canonical3', 'canonical5_multiscale') if scene == 'canonical' else (PRESETS[scene],)
    if feat['preset'] not in allowed_presets or feat['version'] != 1:
        raise ValueError('Feature preset/version does not match scenario')
    if feat['preset'] == 'canonical5_multiscale':
        scales = np.asarray([feat.get('time_scale', np.nan), feat.get('tau_scale', np.nan)], dtype=float)
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('canonical5_multiscale requires finite positive time_scale and tau_scale')
    if model['input_dim'] != len(FEATURES[feat['preset']]) or model['output_dim'] != 1:
        raise ValueError('MLP shape does not match Phi')
    if model['activation'] != 'tanh' or min(model['hidden_dim'],model['num_hidden_layers']) < 1:
        raise ValueError('PEAS requires a positive-width tanh MLP')
    if c['runtime']['dtype'] != 'float32' or c['runtime']['threads'] < 1:
        raise ValueError('The retained numerical kernels require float32 and positive threads')
    nondim = c['problem']['nondimensionalization']
    if min(nondim['length_scale'], nondim['diffusivity']) <= 0:
        raise ValueError('Nondimensionalization scales must be positive')
    g = c['problem']['domain']
    if float(g.get('rx_radius', g.get('r_min',1))) != 1.0:
        raise ValueError('Dimensionless RX radius must be one')
    if scene in ('concentric','eccentric'):
        d = np.linalg.norm(np.asarray(g['rx_center'])-np.asarray(g['outer_center']))
        if not d + g['rx_radius'] < g['outer_radius']:
            raise ValueError('RX must be strictly inside the outer sphere')
        if (scene == 'concentric') != (d < 1e-10):
            raise ValueError('Scene name and sphere centers disagree')
        validate_sphere_config(native_config(c))
    if scene == 'pipe':
        if g['z_min'] >= g['z_max'] or np.hypot(*g['rx_center'][:2])+1 >= g['pipe_radius']:
            raise ValueError('Invalid pipe geometry')
        if not g['z_min']+1 < g['rx_center'][2] < g['z_max']-1:
            raise ValueError('RX must be inside pipe caps')
        expected = {'mode':'augmented','spatial_transform':'mirror_log','tau_transform':'log1p',
                    'sqrt_time_transform':'smooth','include_arrival':True,'include_tau':True}
        if any(feat.get(k) != v for k,v in expected.items()):
            raise ValueError('symmetric11_arrival settings do not match its declared version')
        if g['rx_center'][1] != 0 or g['rx_center'][2] != (g['z_min']+g['z_max'])/2:
            raise ValueError('Pipe mirror features require receiver on both symmetry planes')
    if scene == 'canonical' and not feat.get('include_tau', True):
        raise ValueError('Canonical features require tau')
    methods = {'canonical':('radial_fem',),'concentric':('radial_fem',),
               'eccentric':('bispherical_fem',),'pipe':('pipe_fem',)}[scene]
    if c['steady_state']['method'] not in methods:
        raise ValueError('Steady provider does not match scene')
    if scene=='concentric' and c['steady_state']['method']=='radial_fem':
        params = c['steady_state']['parameters']
        count, power = params.get('num_elements'), params.get('mesh_power')
        if (type(count) is not int or count < 2 or not isinstance(power,(int,float))
                or not np.isfinite(power) or power <= 0
                or not np.isfinite(c['problem']['boundary']['kappa']) or c['problem']['boundary']['kappa'] < 0):
            raise ValueError('Concentric FEM needs >=2 elements, positive mesh_power and nonnegative Robin kappa')
    tc = c['problem']['time']
    if not 0 < tc['t_min_pde'] < tc['t_max']:
        raise ValueError('Invalid PDE time interval')
    tr = c['training']
    if 'without_late_loss' in tr:
        if scene != 'pipe' or type(tr['without_late_loss']) is not bool:
            raise ValueError('without_late_loss must be a boolean for the pipe scenario')
    objective = tr.get('objective', 'original')
    if objective not in ('original', 'e04_relative'):
        raise ValueError('Unknown physics training objective')
    if objective == 'e04_relative':
        if scene != 'canonical' or feat['preset'] != 'canonical5_multiscale':
            raise ValueError('E04 requires canonical5_multiscale')
        if c['sampling'].get('canonical_mode') != 'stratified':
            raise ValueError('E04 requires its stratified CPU sampler')
        if c['steady_state']['parameters']['outer_radius'] != 200.0:
            raise ValueError('E04 distance groups cover the FEM shell r=1..200')
        if tr.get('lbfgs',{}).get('enabled') or tr.get('adaptive_sampling',{}).get('enabled'):
            raise ValueError('E04 uses Adam with its stateful stratified sampler')
    elif scene == 'canonical' and c['sampling'].get('canonical_mode', 'original') != 'original':
        raise ValueError('Stratified canonical sampling requires the E04 objective')
    if tr['epochs'] < 1 or min(tr['batch_sizes'].values()) < 1:
        raise ValueError('Positive epochs and batch sizes required')
    stages = tr.get('lr_schedule') or [{'epochs':tr['epochs'],'lr':tr['lr']}]
    if sum(s['epochs'] for s in stages) != tr['epochs'] or any(s['epochs']<1 or s['lr']<=0 for s in stages):
        raise ValueError('Adam schedule must cover exactly training.epochs')
    if tr.get('post_lbfgs_adam',{}).get('enabled'):
        raise ValueError('The selected four formal configurations do not use post-LBFGS Adam')
    if not c['evaluation']['raw_predictions']:
        raise ValueError('Unified evaluation records raw predictions')
    return c

def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as f:
        if path.suffix.lower() == '.json':
            c = json.load(f)
        elif path.suffix.lower() in ('.yaml','.yml'):
            c = yaml.safe_load(f)
        else:
            raise ValueError('Configuration must use .json, .yaml or .yml')
    if not isinstance(c,dict):
        raise ValueError('Configuration must be a mapping/object')
    roots = {'PEAS_ROOT':str(repository_root())}
    roots.update(os.environ)
    def resolve(value):
        if value is None: return None
        def substitute(m):
            if m[1] not in roots: raise ValueError(f'Undefined path variable {m[1]}')
            return roots[m[1]]
        value = re.sub(r'\$\{([^}]+)\}',substitute,str(value))
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent/p).resolve())
    c['paths']['output_root'] = resolve(c['paths']['output_root'])
    c['steady_state']['checkpoint'] = resolve(c['steady_state'].get('checkpoint'))
    c['evaluation']['reference'] = resolve(c['evaluation'].get('reference'))
    return validate_config(c)

def native_config(c, output=None):
    """Internal bridge to the retained sampling/residual arithmetic, not old programs."""
    c = copy.deepcopy(c)
    n = {'project':{'seed':c['seed']},'domain':c['problem']['domain'],
         'boundary':c['problem']['boundary'],'time':c['problem']['time'],
         'features':c['features'],'model':c['model'],'runtime':c['runtime'],
         'training':c['training'],'u_training':c['training']}
    n.update(c['sampling'])
    n.update(c['residual'])
    for key in ('adaptive_sampling','lbfgs','validation','post_lbfgs_adam'):
        n[key] = c['training'].get(key,{'enabled':False})
    params = c['steady_state']['parameters']
    if c['scenario']=='canonical': n['q_infty'] = params
    if c['scenario']=='eccentric': n['q_fem'] = params
    if c['scenario']=='pipe':
        n['mesh'] = params['mesh']
        n['fem_q_infty'] = {**params,'checkpoint':c['steady_state'].get('checkpoint')}
    if output is not None:
        n['outputs'] = {f'{name}_dir':str(Path(output)/name) for name in ('checkpoints','reports','plots','meshes')}
    return n

def save_config(c, path):
    from peas.common.checkpoints import portable_config
    c=portable_config(c,Path(path).parent)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    path=Path(path)
    if path.suffix.lower() == '.json':
        text=json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    elif path.suffix.lower() in ('.yaml','.yml'):
        text=yaml.safe_dump(c,sort_keys=False)
    else:
        raise ValueError('Configuration must use .json, .yaml or .yml')
    path.write_text(text,encoding='utf-8')
