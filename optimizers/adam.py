from .optimizer import Optimizer

class Adam(Optimizer):
    """Adam optimizer with learning rate decay"""
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, decay=0.0):
        super().__init__(params, lr, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.v = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * xp.square(param.grad.data)

            # Bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
