"""Same tanh MLP family; each ablation changes one declared design axis."""
from __future__ import annotations

import numpy as np
import torch

from peas.common.config import native_config
from peas.common.modules.features import FeatureConstructor
from peas.common.modules.input_adapter import InputAdapter
from peas.common.modules.steady import FixedSteadyState
from peas.common.modules.transient import TransientMLP


class AblationModel:
    def __init__(self, config, device):
        from peas.ablations.physics import TrainingPhysics
        self.config = config
        self.variant = config["ablation"]["variant"]
        self.device = torch.device(device)
        self.input_adapter = InputAdapter(config)
        self.features = FeatureConstructor(config, native_config(config))
        self.transient = TransientMLP({**config["model"],
                                     "input_dim": config["ablation"]["network_input_dim"]}).to(device)
        self.steady = None if self.variant == "no_steady" else FixedSteadyState(config, device)
        self.source_q = None
        if self.variant == "no_reformulation":
            source = torch.tensor([config["ablation"]["source"]], dtype=torch.float32, device=device)
            coordinates = self.input_adapter.native_coordinates(source)
            self.source_q = self.steady.value_and_derivative(coordinates)[0].detach()
            if not torch.isfinite(self.source_q).all() or self.source_q.item() <= 0:
                raise ValueError("Forward source needs a positive finite steady absorption probability")
        self.training = TrainingPhysics(config)

    def latent(self, t, coordinates):
        if self.variant == "no_features":
            from peas.ablations.no_features import features
            phi = features(t, coordinates, self.config)
        else:
            phi = self.features(t, coordinates)
        return self.transient(phi)

    def field(self, t, coordinates):
        """Backward q, or forward density p; never clip probabilities/densities."""
        value = self.latent(t, coordinates)
        if self.variant == "no_steady":
            return value
        if self.variant == "no_reformulation":
            return self.source_q * value
        q, _ = self.steady.value_and_derivative(coordinates)
        return q * value

    def predict(self, times, points, *, batch_size=4096):
        times = np.asarray(times, dtype=float)
        points = self.input_adapter.validate_points(points)
        if times.ndim != 1 or not len(times) or np.any(times < 0) or not np.isfinite(times).all():
            raise ValueError("times must be a finite nonnegative vector")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.variant == "no_reformulation":
            from peas.ablations.no_reformulation import absorption_curve
            source = np.asarray(self.config["ablation"]["source"])
            if len(points) != 1 or not np.allclose(points[0], source, atol=1e-10, rtol=0):
                raise ValueError("A fixed-source forward model can only predict its own source curve")
            return absorption_curve(self, times, batch_size=batch_size)["prediction"][None]
        previous = self.transient.training
        self.transient.eval()
        result = []
        try:
            with torch.no_grad():
                # Match the formal pipe predictor's double-coordinate FEM lookup.
                q_pipe = None
                if self.config["scenario"] == "pipe" and self.steady is not None:
                    p = self.config["steady_state"]["parameters"]
                    q_pipe, _, fallback = self.steady.provider.provider.value_and_grad(
                        points, candidates=p["locator_candidates"])
                    if np.any(fallback):
                        raise ValueError("FEM locator fallback at prediction points")
                for i, point in enumerate(points):
                    values = []
                    for start in range(0, len(times), batch_size):
                        ts = times[start:start + batch_size]
                        t = torch.tensor(ts[:, None], dtype=torch.float32, device=self.device)
                        xyz = torch.tensor(np.repeat(point[None], len(ts), axis=0),
                                           dtype=torch.float32, device=self.device)
                        coordinates = self.input_adapter.native_coordinates(xyz)
                        if q_pipe is not None:
                            value = float(q_pipe[i, 0]) * self.latent(t, coordinates)
                        else:
                            value = self.field(t, coordinates)
                        values.append(value.cpu().numpy().ravel())
                    result.append(np.concatenate(values))
        finally:
            self.transient.train(previous)
        result = np.asarray(result)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite prediction")
        return result

