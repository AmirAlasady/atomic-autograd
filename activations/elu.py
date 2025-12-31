from atomic.core.autograd.tensor import Tensor
from .activation import Activation

class ELU(Activation):
    def __init__(self, alpha=1.0, device='cpu'):
        super().__init__(device)
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        
        condition = inputs > 0
        # inputs.where(condition, else_val)
        # ELU: x if x > 0 else alpha * (exp(x) - 1)
        return inputs.where(condition, self.alpha * (inputs.exp() - 1))

    def state_dict(self):
        return {"activation": type(self).__name__, "alpha": self.alpha}

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get("alpha", self.alpha)

    def __call__(self, x):
        return self.forward(x)
