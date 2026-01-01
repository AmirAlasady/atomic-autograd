from atomic.core.autograd.tensor import Tensor
from .activation import Activation

class ReLU(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        return inputs.relu()

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)
