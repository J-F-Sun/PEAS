"""Deterministic feature registry and scene dispatch."""
from peas.common.sphere_features import FEATURE_NAMES, sphere_input_scale, validate_sphere_config, sphere_features
from peas.scenarios.canonical.features import canonical_features, canonical_multiscale_values
from peas.scenarios.pipe.features import pipe_features, drift_arrival_feature

FEATURES = {
    "canonical3": ("t", "1/d", "t/d^2"),
    "canonical5_multiscale": ("t/T", "1/d", "t/d^2", "log1p(t)/log1p(T)",
        "log1p((t/d^2)/s)/log1p(T/s)"),
    "axis10_arrival": FEATURE_NAMES,
    "symmetric11_arrival": ("t/T", "x/R", "log1p(y^2/R^2)", "log1p(z^2/L^2)",
        "(d-a)/R", "(R-rho_eps)/R", "gap_caps/H", "smooth_sqrt(t)/sqrt(T)",
        "1/d", "log1p(t/d^2)", "erfc(gap_rx/sqrt(4*(t+eps)))"),
}


class FeatureConstructor:
    """Deterministic Phi_G. No parameters; coordinates stay in autograd."""
    def __init__(self, config, native):
        self.preset = config["features"]["preset"]
        self.names = FEATURES[self.preset]
        self.native = native
        self.function = {"canonical3": canonical_features,
            "canonical5_multiscale": canonical_features,
            "axis10_arrival": sphere_features,
            "symmetric11_arrival": pipe_features}[self.preset]

    def __call__(self, t, coordinates):
        values = self.function(t, coordinates, self.native)
        if values.shape != (t.shape[0], len(self.names)):
            raise ValueError("Phi output shape differs from declared feature order")
        return values
