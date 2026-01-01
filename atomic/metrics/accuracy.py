import numpy as np
from atomic.core.autograd.tensor import Tensor

try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None

class Accuracy:
    """Flexible accuracy metric supporting multiple formats"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> float:
        """
        Args:
            pred: (batch_size, num_classes) logits/probabilities
            target: (batch_size,) class indices or (batch_size, num_classes) one-hot

        Returns:
            Accuracy score between 0 and 1
        """
        # Get prediction classes
        pred_classes = pred.argmax(axis=-1).data

        # Handle different target formats
        if target.ndim > 1 and target.shape[-1] > 1:
            # One-hot encoded targets
            target_classes = target.argmax(axis=-1).data
        else:
            # Class indices
            target_classes = target.data.squeeze()

        # Device-aware comparison
        if isinstance(pred_classes, np.ndarray):
            return np.mean(pred_classes == target_classes)
        else:  # cupy
            return float(cp.mean(pred_classes == target_classes))
