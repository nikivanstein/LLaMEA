import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None
        self.pswarm = np.zeros((self.dim,))
        self.pswarm_best = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Global Exploration using Differential Evolution
            f_x = func(self.x)
            if f_x < self.f_best:
                self.f_best = f_x
                self.x_best = self.x.copy()

            # Local Exploration using Particle Swarm Optimization
            if np.random.rand() < 0.2:
                # Update swarm positions
                self.pswarm = self.pswarm + np.random.uniform(-1.0, 1.0, size=(self.dim,))
                self.pswarm = (self.pswarm + self.x) / 2

                # Update swarm velocities
                self.pswarm = self.pswarm * 0.9 + np.random.uniform(-0.1, 0.1, size=(self.dim,))

                # Update swarm positions
                self.pswarm = self.pswarm + self.pswarm * 0.1

                # Evaluate swarm positions
                f_x_p = func(self.pswarm)
                if f_x_p < self.pswarm_best:
                    self.pswarm_best = f_x_p

                # Update the current position
                self.x = self.pswarm.copy()

            # Update the current position
            self.x = (self.x + np.random.uniform(-1.0, 1.0, size=(self.dim,))) / 2

    def get_best_x(self):
        return self.x_best

    def get_best_f(self):
        return self.f_best

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_DEPSO = HybridDEPSO(budget=100, dim=10)
hybrid_DEPSO()
print(hybrid_DEPSO.get_best_x())
print(hybrid_DEPSO.get_best_f())