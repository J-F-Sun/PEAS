"""Native BoundaryCompliantAnsatz -> SVD -> native Diffusion_Solver.fit."""
import time
import numpy as np
from swimpde.Ansatz import BoundaryCompliantAnsatz
from swimpde.Solver.diffusion_solver import Diffusion_Solver
from .adapters import RadialAnsatz,MixedBoundary
from .geometry import sample_domain,initial_density,sigma_for,normalization,sample_uniform,sphere_surface,contains
from .model import FrozenModel


def fit(c,source):
    source=np.asarray(source,dtype=float);sigma=sigma_for(source,c)
    started=time.perf_counter()
    rng=np.random.default_rng(c['seed']);domain=sample_domain(c,source,sigma,rng)
    sampled=time.perf_counter()
    p0=lambda x:initial_density(x,source,sigma,c)
    cls=RadialAnsatz if c['representation']=='radial' else BoundaryCompliantAnsatz
    ansatz=cls(n_inner_basis=c['model']['inner_basis'],n_outer_basis=c['model']['outer_basis'],
        activation='tanh',parameter_sampler='tanh',target_gen='tanh',domain_margin_percent=0,
        random_state=c['seed'],regularization_scale=c['numerics']['lstsq_rcond'])
    solver=Diffusion_Solver(domain=domain,ansatz=ansatz,u0=p0,boundary_condition=MixedBoundary(domain,c),
        forcing=lambda x,t:np.zeros(len(x)),regularization_scale=c['numerics']['lstsq_rcond'],ode_solver=c['numerics']['ode_method'])
    # Instrument the original setup without changing its mathematical operations.
    timings={}
    native_init=ansatz.init_model
    def timed_ansatz(*args,**kwargs):
        t=time.perf_counter();result=native_init(*args,**kwargs);timings['basis_boundary_seconds']=time.perf_counter()-t;return result
    ansatz.init_model=timed_ansatz
    native_matrices=solver._init_matrices
    def timed_matrices(*args,**kwargs):
        t=time.perf_counter();result=native_matrices(*args,**kwargs);timings['svd_matrices_seconds']=time.perf_counter()-t;return result
    solver._init_matrices=timed_matrices
    solver,status=solver.fit(t_span=[0,c['time']['end']],rtol=c['numerics']['rtol'],atol=c['numerics']['atol'],
        svd_cutoff=c['numerics']['svd_cutoff'],outer_basis=True,init_cond=None)
    elapsed=time.perf_counter()-started
    if status!=0:raise RuntimeError(f'Native ODE solver status {status}')
    model=FrozenModel(solver,c,source,sigma)
    t=c['time']['end'];state=model.coefficients(np.array([0,t]))
    if not np.isfinite(state).all():raise FloatingPointError('Nonfinite ODE coefficients')
    ic=model.density(domain.interior_points,[0])[:,0];target=p0(domain.interior_points)
    boundary=model.basis(domain.boundary_points)
    dn=model.basis(domain.boundary_points,domain.normal_vectors)
    bc=solver.boundary_condition
    residual=bc.alpha[:,None]*boundary+bc.beta[:,None]*dn
    times=np.linspace(0,t,17);values=residual@model.coefficients(times)
    diagnostics={'sigma':sigma,'rank':len(model.v),'initial_projection_rel_l2':float(np.linalg.norm(ic-target)/max(np.linalg.norm(target),1e-30)),
        'boundary_rms_training_points':float(np.sqrt(np.mean(values**2))),
        'coefficient_max_abs':float(np.max(np.abs(state))),'sampling_seconds':sampled-started,
        'fit_seconds':elapsed,**timings,'nfev':int(solver._fit_info['nfev']),'njev':int(solver._fit_info['njev']),'nlu':int(solver._fit_info['nlu']),
        'ode_seconds':solver._fit_info['seconds'],'ode_status':status}
    probe=domain.interior_points[:8]
    native=solver.evaluate(probe,times)
    diagnostics['checkpoint_native_max_abs_difference']=float(np.max(np.abs(native-model.density(probe,times))))
    native_gradient=solver.ansatz.evaluate_model_gradient(probe)
    direction=np.ones_like(probe)
    native_dn=np.einsum('nkd,nd->nk',native_gradient,direction)@solver._V_a.T
    diagnostics['checkpoint_native_gradient_max_abs_difference']=float(np.max(np.abs(native_dn-model.basis(probe,direction))))
    # Independent physical diagnostics, never used as fitting targets.
    import copy
    independent=copy.deepcopy(c);independent['sampling']['interior_points']=64
    independent['sampling']['surface_points']=max(258,min(c['sampling']['surface_points']+258,2048))
    audit=sample_domain(independent,source,sigma,np.random.default_rng(c['seed']+917))
    ab=MixedBoundary(audit,c)
    basis=model.basis(audit.boundary_points);normal=model.basis(audit.boundary_points,audit.normal_vectors)
    independent_residual=(ab.alpha[:,None]*basis+ab.beta[:,None]*normal)@model.coefficients(times)
    diagnostics['boundary_rms_independent_points']=float(np.sqrt(np.mean(independent_residual**2)))
    g=c['geometry'];rng=np.random.default_rng(c['seed']+1991)
    if c['representation']=='radial':
        nodes,weights=np.polynomial.legendre.leggauss(1024)
        radius=g.get('outer_radius',c['numerics']['outer_cutoff']);a=g['rx_radius']
        xx=(a+(nodes+1)*(radius-a)/2)[:,None]
        weights=weights*(radius-a)/2*4*np.pi*xx[:,0]**2
        mass_basis=weights@model.basis(xx)
        if c['scene']=='concentric':
            out_vector=g['kappa']*4*np.pi*radius**2*model.basis(np.array([[radius]]))[0]
        else:
            out_vector=-g['diffusivity']*4*np.pi*radius**2*model.basis(np.array([[radius]]),np.ones((1,1)))[0]
        outer_rate=out_vector@model.coefficients(times)
        mass=mass_basis@model.coefficients(times)
        diagnostics['mass_quadrature']='1024-point Gauss-Legendre with 4*pi*r^2 measure'
    else:
        # Mixture importance integration resolves both the narrow IC and spread density.
        # This is an observation quadrature, not an ansatz or training change.
        nq=4096;uniform=sample_uniform(rng,nq//2,c);local=[];count=0;spread=sigma*c['sampling']['source_spread']
        while count<nq//2:
            proposal=rng.normal(source,spread,(max(64,2*(nq//2-count)),3));v=proposal[contains(proposal,c)]
            local.append(v);count+=len(v)
        xx=np.vstack((uniform,np.concatenate(local)[:nq//2]))
        volume=(np.pi*g['pipe_radius']**2*(g['z_max']-g['z_min']) if c['scene']=='pipe' else 4*np.pi*g['outer_radius']**3/3)-4*np.pi*g['rx_radius']**3/3
        broad=np.exp(-np.sum((xx-source)**2,axis=1)/(2*spread**2))/((2*np.pi*spread**2)**1.5*normalization(source,spread,c))
        weights=1/(nq*(.5/volume+.5*broad));basis_q=model.basis(xx)
        mass=weights@basis_q@model.coefficients(times)
        diagnostics['mass_quadrature']='4096-point uniform/Gaussian importance estimate; approximate, not an exact conservation certificate'
        diagnostics['density_min_audit']=float(np.min(basis_q@model.coefficients(times)))
        # Flux through all non-RX exits, evaluated with analytic normals.
        outer=audit.labels=='outer';caps=audit.labels=='cap'
        if c['scene'] in ('concentric', 'eccentric'):
            area=4*np.pi*g['outer_radius']**2
        else:
            area=2*np.pi*g['pipe_radius']*(g['z_max']-g['z_min'])
        out_vector=g['kappa']*area*basis[outer].mean(axis=0)
        if caps.any():out_vector-=g['diffusivity']*2*np.pi*g['pipe_radius']**2*normal[caps].mean(axis=0)
        outer_rate=out_vector@model.coefficients(times)
    diagnostics['mass_initial_estimate']=float(mass[0])
    diagnostics['mass_final_estimate']=float(mass[-1])
    diagnostics['outer_loss_rate_min_audit']=float(outer_rate.min())
    diagnostics['snapshot_density_min']=float(np.min(model.density(domain.interior_points[:128],times)))
    # Integrate exits at dense times; mass is independently integrated in space.
    from scipy.integrate import cumulative_trapezoid
    dense_times=np.linspace(0,t,c['evaluation']['integration_points'])
    exits=(model.rx_vector()+out_vector)@model.coefficients(dense_times)
    total_loss=np.interp(times,dense_times,cumulative_trapezoid(exits,dense_times,initial=0))
    balance=mass+total_loss
    diagnostics['mass_balance_max_from_one']=float(np.max(np.abs(balance-1)))
    diagnostics['mass_balance_max_from_initial_estimate']=float(np.max(np.abs(balance-mass[0])))
    diagnostics['density_min_audit']=diagnostics.get('density_min_audit',diagnostics['snapshot_density_min'])
    return model,diagnostics
