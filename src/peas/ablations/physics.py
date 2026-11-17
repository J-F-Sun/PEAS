"""Scene samplers and physics objectives shared by the three ablations."""
from __future__ import annotations

import copy
from importlib import import_module
import torch
from torch import nn

from peas.common.config import native_config
from peas.common.utils import grad_scalar
from peas.scenarios.canonical.physics import CPUSampler, GroupedObjective
from peas.scenarios.pipe import physics as pipe


class RawFeatureAdapter(nn.Module):
    """Retain the E04 [t,y,t*y²] autograd graph before applying the chosen Phi."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def out(self):
        return self.model.transient.net[-1]

    def forward(self, raw):
        # With engineered canonical features, preserve the original use of tau.
        if self.model.variant != "no_features":
            from peas.scenarios.canonical.features import canonical_multiscale_values
            phi = canonical_multiscale_values(raw[:, :1], raw[:, 1:2], raw[:, 2:3],
                                              self.model.config["features"])
            return self.model.transient(phi)
        return self.model.latent(raw[:, :1], raw[:, 1:2])


class DirectCanonicalObjective(GroupedObjective):
    """Same E04 learned-response/group/gradient policy, on the ablated unknown."""
    def __call__(self, model, batch):
        from peas.ablations import no_reformulation, no_steady
        module = no_steady if model.variant == "no_steady" else no_reformulation
        residuals, response = module.canonical_parts(model, batch)
        original = {k: v.square().mean() for k, v in residuals.items()}
        ids = self.ids(batch["pde"][:, 1:])
        ic_ids = self.ids(batch["ic"][:, 1:])
        with torch.no_grad():
            for k in range(6):
                selected = response.detach().flatten()[ids == k]
                if selected.numel():
                    self.amplitude_ema[k] = .99 * self.amplitude_ema[k] + .01 * selected.square().mean().cpu()
        blend = min(1., max(0., (self.epoch - 2500) / 1000.))
        scales = self.amplitude_ema.to(response).sqrt().clamp(.05, 1.)
        grouped = {"pde": self.grouped_mean(residuals["pde"].flatten(), ids, scales),
                   "ic": self.grouped_mean(residuals["ic"].flatten(), ic_ids, scales),
                   "bc_rx": original["bc_rx"], "bc_far": original["bc_far"]}
        mixed = {k: (1 - blend) * original[k] + blend * grouped[k] for k in original}
        if blend and self.epoch % 100 == 0:
            norms = []
            for value in mixed.values():
                grads = torch.autograd.grad(value, list(model.transient.net[-1].parameters()),
                                            retain_graph=True, allow_unused=True)
                norms.append(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None) + 1e-16))
            observed = torch.stack(norms).cpu()
            self.gradient_ema = .95 * self.gradient_ema + .05 * observed
            target = self.gradient_ema[:3].clamp_min(1e-8).log().mean().exp()
            self.factors = (target / self.gradient_ema.clamp_min(1e-8)).clamp(.2, 5.)
        return {k: value * ((1 - blend) + blend * self.factors[i].to(value))
                for i, (k, value) in enumerate(mixed.items())}


def factorized_pipe_losses(model, batch):
    """Formal pipe loss arithmetic, with the ablated feature constructor."""
    t = batch["pde_t"].clone().requires_grad_(True)
    xyz = batch["pde_xyz"].clone().requires_grad_(True)
    u = model.latent(t, xyz)
    lap, gradient = pipe.laplacian(u, xyz)
    q, grad_q = model.steady.provider.value_and_grad(xyz)
    residual = q * (grad_scalar(u, t) - lap) - 2 * (grad_q * gradient).sum(dim=1, keepdim=True)
    hybrid = model.config["residual"]["hybrid_residual"]
    if hybrid["enabled"]:
        residual = residual / (q + hybrid["epsilon_q"])
    ui = model.latent(batch["ic_t"], batch["ic_xyz"])
    ur = model.latent(batch["rx_t"], batch["rx_xyz"])
    wall = batch["wall_xyz"].clone().requires_grad_(True)
    uw = model.latent(batch["wall_t"], wall)
    rho = torch.linalg.vector_norm(wall[:, :2], dim=1, keepdim=True)
    normal = torch.cat((wall[:, :2] / rho, torch.zeros_like(rho)), dim=1)
    rw = (grad_scalar(uw, wall) * normal).sum(dim=1, keepdim=True)
    uc = model.latent(batch["caps_t"], batch["caps_xyz"])
    qc, _ = model.steady.provider.value_and_grad(batch["caps_xyz"])
    q_hat = q * u
    result = {"pde": residual.square().mean(), "ic": ui.square().mean(),
              "rx": (ur - 1).square().mean(), "wall": rw.square().mean(),
              "caps": (qc * uc).square().mean(),
              "range": (torch.relu(-u).square() + torch.relu(u - 1.2).square() +
                        torch.relu(-q_hat).square() + torch.relu(q_hat - 1).square()).mean()}
    if "late_t" in batch:
        result["late"] = (model.latent(batch["late_t"], batch["late_xyz"]) - 1).square().mean()
    return result


class TrainingPhysics:
    def __init__(self, config):
        self.config = config
        self.native = native_config(config)
        self.variant = config["ablation"]["variant"]
        self.scene = config["scenario"]
        self.generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 100000)
        self.adaptive_points = None
        if self.scene in ("concentric", "eccentric"):
            self.kernel = import_module(f"peas.scenarios.{self.scene}.physics")
        if self.scene == "canonical":
            self.sampler = CPUSampler(self.native, {"seed": config["seed"], "sampling": "stratified"})
            objective_cls = GroupedObjective if self.variant == "no_features" else DirectCanonicalObjective
            self.objective = objective_cls(self.native, "relative")

    def make_batch(self, device, *, sizes=None, adaptive_points=None):
        if adaptive_points is not None and self.scene not in ("concentric", "eccentric"):
            raise ValueError("This formal scenario has no residual-adaptive sampler")
        sizes = sizes or self.config["training"]["batch_sizes"]
        # An independent CPU stream keeps sampling independent of MLP input size.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            if self.scene == "canonical":
                batch = self.sampler.batch(torch.device("cpu"), sizes)
            elif self.scene in ("concentric", "eccentric"):
                adaptive = adaptive_points if adaptive_points is not None else self.adaptive_points
                if adaptive is not None:
                    adaptive = tuple(p.cpu() for p in adaptive)
                batch = self.kernel.make_batch(self.native, torch.device("cpu"),
                                               batch_sizes=sizes, adaptive_points=adaptive)
            else:
                batch = pipe.make_batch(self.native, torch.device("cpu"), sizes)
            if self.variant == "no_reformulation":
                from peas.ablations.no_reformulation import augment_batch
                batch = augment_batch(batch, self.config)
            self.generator.set_state(torch.get_rng_state())
        return {k: v.to(device) for k, v in batch.items()}

    def losses(self, model, batch):
        if self.scene in ("concentric", "eccentric"):
            from peas.ablations.spheres import losses
            return losses(model, batch)
        if self.scene == "canonical":
            if self.variant == "no_features":
                native = copy.deepcopy(self.native)
                native["features"]["preset"] = "canonical3"
                return self.objective(RawFeatureAdapter(model), batch, native, model.steady.provider)
            return self.objective(model, batch)
        if self.config["training"].get("without_late_loss", False):
            from peas.scenarios.pipe_wo_late_loss import without_late_points
            batch = without_late_points(batch)
        if self.variant == "no_features":
            return factorized_pipe_losses(model, batch)
        if self.variant == "no_steady":
            from peas.ablations.no_steady import pipe_losses
        else:
            from peas.ablations.no_reformulation import pipe_losses
        return pipe_losses(model, batch)

    def set_epoch(self, epoch):
        if self.scene == "canonical":
            self.objective.epoch = int(epoch)

    def maybe_refresh_adaptive(self, model, device, epoch):
        """One-based epoch, matching the formal sphere training schedule."""
        cfg = self.config["training"].get("adaptive_sampling", {})
        start = int(cfg.get("start_epoch", 800))
        if (self.scene in ("concentric", "eccentric") and cfg.get("enabled")
                and epoch >= start and (self.adaptive_points is None or
                                       (epoch - start) % int(cfg["refresh_every"]) == 0)):
            from peas.ablations.spheres import refresh_adaptive
            self.adaptive_points = refresh_adaptive(model, device, self.generator)

    def seeded_batch(self, device, sizes, seed):
        c = copy.deepcopy(self.config)
        c["seed"] = int(seed)
        return TrainingPhysics(c).make_batch(device, sizes=sizes)

    def state_dict(self):
        state = {"rng": self.generator.get_state()}
        if self.scene == "canonical":
            state.update(sampler_rng=self.sampler.generator.get_state(), objective=self.objective.state_dict())
        if self.scene in ("concentric", "eccentric"):
            state["adaptive_points"] = self.adaptive_points
        return state

    def load_state_dict(self, state):
        self.generator.set_state(state["rng"].cpu())
        if self.scene in ("concentric", "eccentric"):
            points = state.get("adaptive_points")
            self.adaptive_points = tuple(p.cpu() for p in points) if points is not None else None
        if self.scene == "canonical":
            self.sampler.generator.set_state(state["sampler_rng"].cpu())
            self.objective.load_state_dict({k: v.cpu() if torch.is_tensor(v) else v
                                           for k, v in state["objective"].items()})

