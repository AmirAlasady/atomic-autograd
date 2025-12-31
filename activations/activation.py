from abc import ABC, abstractmethod
from atomic.core.basestructure.base import Base_Layer
from atomic.core.autograd.tensor import Tensor

class Activation(Base_Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def set_gpu(self):
        # Consider using 'cuda' if that's what your Tensor expects.
        self.device = 'cuda'

    def set_cpu(self):
        self.device = 'cpu'

    @abstractmethod
    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device, dtype=self.dtype)  # Add dtype
        elif inputs.device != self.device:
            inputs = inputs.to(self.device)
        return self._forward_impl(inputs)
