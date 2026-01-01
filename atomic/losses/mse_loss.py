from atomic.core.autograd.tensor import Tensor
from .loss import Loss

class MSELoss(Loss):
    """Mean Squared Error Loss with automatic broadcasting"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        """
        Args:
            pred: Prediction tensor of shape (batch_size, ...)
            target: Target tensor of shape (batch_size, ...)

        Returns:
            MSE loss tensor with automatic broadcasting
        """
        return (pred - target).square().mean()
