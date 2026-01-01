from atomic.core.autograd.tensor import Tensor
from .activation import Activation

class Softmax(Activation):
    def __init__(self,axis=-1, device='cpu'):
        super().__init__(device)
        self.axis = axis
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        # Optionally, add an axis parameter if needed:
        # self.axis = axis

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        # Currently using axis=0; adjust if you want to apply softmax over a different axis.
        return inputs.softmax(axis=self.axis)

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)
