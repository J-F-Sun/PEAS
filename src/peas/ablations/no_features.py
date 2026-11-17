"""Remove engineered Phi while retaining nondimensional coordinate scaling."""
import torch


def features(t, coordinates, config):
    if config["scenario"] == "canonical":
        # y=1/r is the native PDE coordinate, set by InputAdapter, not an added input.
        return torch.cat((t / config["features"]["time_scale"], coordinates), dim=1)
    g, f = config["problem"]["domain"], config["features"]
    if config["scenario"] in ("concentric", "eccentric"):
        from peas.common.sphere_features import sphere_input_scale
        from peas.common.config import native_config
        scale = sphere_input_scale(native_config(config))
        return torch.cat((t / f["time_scale"],
                          (coordinates - coordinates.new_tensor(g["outer_center"])) / scale), dim=1)
    midpoint = (g["z_min"] + g["z_max"]) / 2
    return torch.cat((t / f["time_scale"], coordinates[:, :2] / g["pipe_radius"],
                      (coordinates[:, 2:3] - midpoint) / f["axial_scale"]), dim=1)

