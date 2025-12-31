from .optimizer import Optimizer

class SGD(Optimizer):
    """Momentum SGD with learning rate decay"""
    def __init__(self, params, lr=0.01, momentum=0.9, decay=0.0):
        super().__init__(params, lr, decay)
        self.momentum = momentum
        self.velocities = [self._get_xp(p).zeros_like(p.data) for p in self.params]

    def step(self):
        self.iterations += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + \
                                (1 - self.momentum) * param.grad.data

            # Update parameters
            param.data -= self.lr * self.velocities[i]
