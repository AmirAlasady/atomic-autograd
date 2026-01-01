from atomic.core.autograd.tensor import Tensor
from .activation import Activation

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        # Fixed implementation: alpha * x for negative values
        # logic is now inside Tensor.leaky_relu() or can be explicit here
        # Original code used element-wise where
        return inputs.leaky_relu(negative_slope=self.alpha)

    def state_dict(self):
        return {"activation": type(self).__name__, "alpha": self.alpha}

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get("alpha", self.alpha)

    def __call__(self, x):
        return self.forward(x)
