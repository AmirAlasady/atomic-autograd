import numpy as np
from atomic.core.autograd.tensor import Tensor
from .loss import Loss

class SoftmaxCrossEntropyLoss(Loss):
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        # If target has the same number of dimensions as pred,
        # assume it's one-hot encoded; otherwise, assume it's indices.
        if target.ndim == pred.ndim:
            class_indices = target.argmax(axis=-1).astype(np.int64)
        else:
            class_indices = target.astype(np.int64)

        # Numerical stability: compute log-softmax
        max_pred = pred.max(axis=-1, keepdims=True)
        log_sum_exp = (pred - max_pred).exp().sum(axis=-1, keepdims=True).log()
        log_softmax = pred - max_pred - log_sum_exp

        # Flatten log_softmax from shape (batch, seq_len, vocab_size) to (batch * seq_len, vocab_size)
        log_softmax_flat = log_softmax.reshape(-1, log_softmax.shape[-1])
        
        # Flatten class indices from shape (batch, seq_len) to (batch * seq_len, 1)
        class_indices = class_indices.reshape(-1, 1)

        # Gather the log probabilities for the correct classes along axis 1
        nll_loss = -log_softmax_flat.gather(1, class_indices)
        return nll_loss.mean()
