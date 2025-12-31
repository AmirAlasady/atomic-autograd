from atomic.core.autograd.tensor import Tensor
from .loss import Loss

class BCELoss(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Binary Cross Entropy with proper Tensor operations"""
        ones = Tensor(1.0)
        epsilon = Tensor(1e-7)

        # Clip using Tensor instances
        clipped = pred.clip(epsilon, ones - epsilon)
        return -(target * clipped.log() + (ones - target) * (ones - clipped).log()).mean()
