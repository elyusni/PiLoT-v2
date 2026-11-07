# Expose the optimizer interface while keeping its implementation in a binary module.
from ._lm_learned_optimizer import DampingNet, LearnedOptimizer as _LearnedOptimizer
from ._lm_learned_optimizer import transform_p3d, project_p3d

class LearnedOptimizer(_LearnedOptimizer):
    """Preserve the public model name for dynamic loading and existing checkpoints."""
