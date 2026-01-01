import numpy as np

try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None

class Optimizer:
    """Base optimizer class with learning rate decay"""
    def __init__(self, params, lr=0.01, decay=0.0):
        """
        Args:
            params: List of trainable parameters (Tensors)
            lr: Initial learning rate
            decay: Learning rate decay factor (per epoch)
        """
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.iterations = 0
        self.xp = np  # Default to numpy

    def _get_xp(self, param):
        """Get correct numerical library for parameter"""
        return np if param.device == 'cpu' else cp # type: ignore

    def step(self):
        """Update parameters (to be implemented by subclasses)"""
        raise NotImplementedError

    def decay_lr(self):
        """Exponential learning rate decay"""
        self.lr *= (1.0 - self.decay)
