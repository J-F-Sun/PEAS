"""Compose the six PEAS components for eccentric."""

def build(config, device):
    from peas.common.config import native_config
    from peas.common.model import PEASModel
    from peas.common.modules.input_adapter import InputAdapter
    from peas.common.modules.features import FeatureConstructor
    from peas.common.modules.transient import TransientMLP
    from peas.common.modules.steady import FixedSteadyState
    from peas.common.modules.ansatz import MultiplicativeAnsatz
    from peas.common.modules.training import PhysicsTraining
    input_adapter = InputAdapter(config)
    features = FeatureConstructor(config, native_config(config))
    transient = TransientMLP(config['model']).to(device)
    steady = FixedSteadyState(config, device)
    ansatz = MultiplicativeAnsatz()
    training = PhysicsTraining(config)
    return PEASModel(config, input_adapter, features, transient, steady, ansatz, training)
