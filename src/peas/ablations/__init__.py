"""Controlled PEAS ablations for all four formal scenarios."""

VARIANTS = ("no_reformulation", "no_steady", "no_features")
SCENES = ("canonical", "concentric", "eccentric", "pipe")
DEFAULT_SCENES = ("canonical", "pipe")
SPHERE_SCENES = ("concentric", "eccentric")


def select_scenes(value):
    """Keep legacy 'both'; permit explicit lists and the new 'spheres' pair."""
    import argparse
    groups = {"both": DEFAULT_SCENES, "spheres": SPHERE_SCENES, "all": SCENES}
    result = groups.get(value, tuple(s.strip() for s in value.split(",")))
    if not result or len(set(result)) != len(result) or any(s not in SCENES for s in result):
        raise argparse.ArgumentTypeError("Use scene names separated by commas, or both/spheres/all")
    return result

