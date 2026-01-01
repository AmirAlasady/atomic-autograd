from atomic.core.autograd.tensor import Tensor
from .loss import Loss

class CrossEntropy(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Numerically stable implementation
        max_val = pred.max(axis=-1, keepdims=True)
        stable_exp = (pred - max_val).exp()
        softmax = stable_exp / stable_exp.sum(axis=-1, keepdims=True)  # Now works!
        return -(target * softmax.log()).mean()
