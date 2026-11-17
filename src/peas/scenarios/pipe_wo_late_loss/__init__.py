"""Pipe ablation removing only the late-time loss from the shared objective."""
from peas.common.modules.training import PhysicsTraining


def without_late_points(batch):
    return {key: value for key, value in batch.items() if key not in ("late_t", "late_xyz")}


class WithoutLateLossTraining(PhysicsTraining):
    """Keep paired sampling identical, but exclude late targets from every loss."""

    def losses(self, model, batch):
        return super().losses(model, without_late_points(batch))
