from atomic.core.autograd.tensor import Tensor

class Loss:
    """Base loss class with numerical stability safeguards"""
    EPSILON = 1e-12

    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        raise NotImplementedError
