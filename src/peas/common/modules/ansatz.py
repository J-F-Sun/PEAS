from dataclasses import dataclass

@dataclass(frozen=True)
class MultiplicativeAnsatz:
    """q_theta=Q*u_theta; output has no probability clipping."""
    def __call__(self, Q, u):
        return Q*u
