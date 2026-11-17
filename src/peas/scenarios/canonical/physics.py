"""E04 canonical sampling and relative physics objective (round 2).

This retains the experimental arithmetic and differentiable graph. No transient
reference values enter training; the fixed radial FEM field is unchanged.
"""
import copy
import math
import torch
from torch import nn
from peas.common.modules.features import canonical_multiscale_values
from peas.common.modules.training import PhysicsTraining
from peas.scenarios.canonical.physics_base import pde_residual, predict_u_y


class CPUSampler:
    """Own CPU generator, independent of model initialization and device."""
    def __init__(self, native, spec):
        self.native, self.spec = native, spec
        self.generator = torch.Generator(device='cpu').manual_seed(spec['seed'] + 100000)

    def rand(self, n):
        return torch.rand((n, 1), generator=self.generator, dtype=torch.float32)

    def times(self, n):
        cfg = self.native['time']
        s = cfg['sampling']
        early_count = int(round(n * s['early_fraction']))
        first = cfg['t_min_pde'] + (s['split_t'] - cfg['t_min_pde']) * self.rand(early_count).pow(s['early_bias_power'])
        last = s['split_t'] + (cfg['t_max'] - s['split_t']) * self.rand(n - early_count).pow(s['late_bias_power'])
        values = torch.cat([first, last])
        return values[torch.randperm(n, generator=self.generator)]

    def positions(self, n, ic=False):
        lower = 1.0 / self.native['q_infty']['outer_radius']
        if self.spec['sampling'] == 'uniform_y':
            y = lower + (1.0 - lower) * self.rand(n)
        else:
            # Half covers the entire original shell; half improves r<=25 coverage.
            whole = n // 2
            log_count = n // 4
            a = lower + (1.0 - lower) * self.rand(whole)
            b = torch.exp(-math.log(25.0) * self.rand(log_count))
            c = 1.0 / (8.0 + 17.0 * self.rand(n - whole - log_count))
            y = torch.cat([a, b, c])
            y = y[torch.randperm(n, generator=self.generator)]
        padding = 1e-4 if ic else 1e-5
        return y.clamp(lower + padding, 1.0 - padding)

    def batch(self, device, sizes=None):
        sizes = sizes or self.native['training']['batch_sizes']
        n = sizes['pde']
        y = self.positions(n)
        result = {'pde': torch.cat([self.times(n), y], dim=1)}
        n = sizes['ic']
        result['ic'] = torch.cat([torch.zeros(n, 1), self.positions(n, ic=True)], dim=1)
        n = sizes['bc_rx']
        result['bc_rx'] = torch.cat([self.times(n), torch.ones(n, 1)], dim=1)
        n = sizes['bc_far']
        result['bc_far'] = torch.cat([self.times(n), torch.full((n, 1), 1.0 / self.native['q_infty']['outer_radius'])], dim=1)
        return {k: v.to(device) for k, v in result.items()}


class GroupedObjective:
    # Bins include the full FEM shell; no radial truncation of the PDE.
    edges=(1.,3.,6.,10.,15.,25.,200.001)

    def __init__(self, native, mode):
        self.mode=mode
        self.scale_floor=float(native.get('response_scale_floor', .05))
        if not 0 < self.scale_floor <= 1: raise ValueError('Invalid response scale floor')
        self.diagnostics=[]
        self.epoch=0
        self.amplitude_ema=torch.ones(6)
        self.gradient_ema=torch.ones(4)
        self.factors=torch.ones(4)

    def state_dict(self):
        return dict(amplitude_ema=self.amplitude_ema,gradient_ema=self.gradient_ema,factors=self.factors,epoch=self.epoch)

    def load_state_dict(self,state):
        self.amplitude_ema=state['amplitude_ema'];self.gradient_ema=state['gradient_ema']
        self.factors=state['factors'];self.epoch=state['epoch']

    def ids(self,y):
        r=1/y.detach().flatten()
        return torch.bucketize(r,torch.as_tensor(self.edges[1:-1],device=y.device,dtype=y.dtype))

    def grouped_mean(self,values,ids,scales):
        result=[]
        for k in range(6):
            selected=values[ids==k]
            if selected.numel():result.append(selected.square().mean()/scales[k].square())
        return torch.stack(result).mean()

    def __call__(self,model,batch,config,provider):
        pde,ic,rx,far=[batch[k] for k in ['pde','ic','bc_rx','bc_far']]
        residual=pde_residual(model,pde[:,:1],pde[:,1:],config,provider)
        u_ic=predict_u_y(model,ic[:,:1],ic[:,1:],config)
        u_rx=predict_u_y(model,rx[:,:1],rx[:,1:],config)
        yf=far[:,1:].clone().requires_grad_(True)
        u_far=predict_u_y(model,far[:,:1],yf,config)
        dy=torch.autograd.grad(u_far,yf,torch.ones_like(u_far),create_graph=True)[0]
        original=dict(pde=residual.square().mean(),ic=u_ic.square().mean(),
            bc_rx=(u_rx-1).square().mean(),bc_far=dy.square().mean())
        ids=self.ids(pde[:,1:]);ic_ids=self.ids(ic[:,1:])
        with torch.no_grad():
            u=predict_u_y(model,pde[:,:1],pde[:,1:],config).detach().flatten()
            for k in range(6):
                selected=u[ids==k]
                if selected.numel():
                    self.amplitude_ema[k]=.99*self.amplitude_ema[k]+.01*selected.square().mean().cpu()
        # Gradually introduce equal-distance groups after the original first stage.
        blend=min(1.,max(0.,(self.epoch-2500)/1000.))
        scales=torch.ones(6,device=residual.device,dtype=residual.dtype)
        if self.mode=='relative':
            # Learned response RMS, not the reference solution. Floor caps inverse
            # squared scale at 1 / scale_floor**2; default preserves the original 400 cap.
            scales=self.amplitude_ema.to(residual).sqrt().clamp(self.scale_floor,1.)
        grouped=dict(pde=self.grouped_mean(residual.flatten(),ids,scales),
            ic=self.grouped_mean(u_ic.flatten(),ic_ids,scales),bc_rx=original['bc_rx'],bc_far=original['bc_far'])
        mixed={k:(1-blend)*original[k]+blend*grouped[k] for k in original}
        if blend and self.epoch%100==0:
            params=list(model.out.parameters())
            norms=[]
            for value in mixed.values():
                grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                norms.append(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)+1e-16))
            observed=torch.stack(norms).cpu()
            self.gradient_ema=.95*self.gradient_ema+.05*observed
            # Use PDE/IC/RX as the shared reference; far BC may be nearly zero.
            target=self.gradient_ema[:3].clamp_min(1e-8).log().mean().exp()
            self.factors=(target/self.gradient_ema.clamp_min(1e-8)).clamp(.2,5.)
        if self.epoch % 100 == 0 or self.epoch == 8999:
            with torch.no_grad():
                rms=self.amplitude_ema.sqrt()
                group_rows=[]
                for k in range(6):
                    rr=residual.flatten()[ids==k]
                    ii=u_ic.flatten()[ic_ids==k]
                    group_rows.append(dict(group=k, r_min=self.edges[k], r_max=self.edges[k+1],
                        ema_rms=float(rms[k]), scale=float(scales[k]),
                        at_floor=bool(rms[k] <= self.scale_floor),
                        pde_count=rr.numel(), ic_count=ii.numel(),
                        pde_mse=float(rr.square().mean()) if rr.numel() else None,
                        ic_mse=float(ii.square().mean()) if ii.numel() else None,
                        pde_normalized=float(rr.square().mean()/scales[k].square()) if rr.numel() else None,
                        ic_normalized=float(ii.square().mean()/scales[k].square()) if ii.numel() else None))
                self.diagnostics.append(dict(epoch=self.epoch+1, blend=blend, groups=group_rows,
                    factors=self.factors.tolist(), gradient_ema=self.gradient_ema.tolist(),
                    original={k:float(v) for k,v in original.items()},
                    mixed={k:float(v) for k,v in mixed.items()}))
        return {k:value*((1-blend)+blend*self.factors[i].to(value)) for i,(k,value) in enumerate(mixed.items())}


class RawCanonicalNetwork(nn.Module):
    """Keep the experimental raw [t,y,t*y²] graph before constructing Phi.

    Inference constructs the same five features in FeatureConstructor. Keeping
    the intermediate concatenation here preserves float32 PDE gradient sums.
    """
    def __init__(self, transient, features):
        super().__init__()
        self.transient = transient
        self.features = features

    @property
    def out(self):
        return self.transient.net[-1]

    def forward(self, raw):
        phi = canonical_multiscale_values(raw[:, :1], raw[:, 1:2], raw[:, 2:3], self.features)
        return self.transient(phi)


class E04CanonicalTraining(PhysicsTraining):
    """Persistent sampler/response/gradient statistics with exact continuation."""
    def __init__(self, config):
        super().__init__(config)
        self.physics_native = copy.deepcopy(self.native)
        # The kernel forms the experimental raw descriptors, then the adapter
        # forms Phi. It still feeds five variables into the transient MLP.
        self.physics_native['features']['preset'] = 'canonical3'
        self.sampler = CPUSampler(self.physics_native,
                                  dict(seed=int(config['seed']), sampling='stratified'))
        self.objective = GroupedObjective(self.physics_native, 'relative')

    def make_batch(self, device, *, sizes=None, adaptive_points=None):
        if adaptive_points is not None:
            raise ValueError('E04 sampling does not use adaptive points')
        return self.sampler.batch(device, sizes)

    def losses(self, model, batch):
        network = RawCanonicalNetwork(model.transient, self.config['features'])
        return self.objective(network, batch, self.physics_native, model.steady.provider)

    def set_epoch(self, epoch):
        self.objective.epoch = int(epoch)

    def state_dict(self):
        return dict(version=1, sampler_rng=self.sampler.generator.get_state(),
                    objective=self.objective.state_dict())

    def load_state_dict(self, state):
        if state.get('version') != 1:
            raise ValueError('E04 resume requires its sampler and objective state')
        self.sampler.generator.set_state(state['sampler_rng'].cpu())
        values = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                  for key, value in state['objective'].items()}
        self.objective.load_state_dict(values)
