import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None
        self.p = 0.02040816326530612
        self.adaptive_p = np.zeros((self.budget,))

    def __call__(self, func):
        for i in range(self.budget):
            # Global Exploration using Differential Evolution
            f_x = func(self.x)
            if f_x < self.f_best:
                self.f_best = f_x
                self.x_best = self.x.copy()
            self.adaptive_p[i] = np.random.rand()
            if np.random.rand() < self.adaptive_p[i]:
                # Local Exploration using Particle Swarm Optimization
                x_p = self.x.copy()
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        x_p[j] = self.x[j] + np.random.uniform(-1.0, 1.0)
                    else:
                        x_p[j] = self.x[j] - np.random.uniform(-1.0, 1.0)
                f_x_p = func(x_p)
                if f_x_p < self.f_best:
                    self.f_best = f_x_p
                    self.x_best = x_p.copy()

            # Update the current position
            self.x = (self.x + np.random.uniform(-1.0, 1.0, size=(self.dim,))) / 2

    def get_best_x(self):
        return self.x_best

    def get_best_f(self):
        return self.f_best

# Example usage:
def func(x):
    return np.sum(x**2)

adaptive_hybrid_DEPSO = AdaptiveHybridDEPSO(budget=100, dim=10)
adaptive_hybrid_DEPSO()
print(adaptive_hybrid_DEPSO.get_best_x())
print(adaptive_hybrid_DEPSO.get_best_f())