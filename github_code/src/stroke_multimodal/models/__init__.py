from .basic_early_fusion import BasicEarlyFusion

try:
    from .interaction_early_fusion import InteractionEarlyFusion
except ImportError:
    InteractionEarlyFusion = None

    __all__ = ["BasicEarlyFusion"]
else:
    __all__ = ["BasicEarlyFusion", "InteractionEarlyFusion"]
