import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None
        self.mutation_prob = 0.02040816326530612

    def __call__(self, func):
        for _ in range(self.budget):
            # Global Exploration using Differential Evolution
            f_x = func(self.x)
            if f_x < self.f_best:
                self.f_best = f_x
                self.x_best = self.x.copy()
            
            # Local Exploration using Particle Swarm Optimization
            if np.random.rand() < 0.2:
                x_p = self.x.copy()
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        x_p[i] = self.x[i] + np.random.uniform(-1.0, 1.0)
                    else:
                        x_p[i] = self.x[i] - np.random.uniform(-1.0, 1.0)
                f_x_p = func(x_p)
                if f_x_p < self.f_best:
                    self.f_best = f_x_p
                    self.x_best = x_p.copy()

            # Update the current position with adaptive mutation
            if np.random.rand() < self.mutation_prob:
                mutation = np.random.uniform(-1.0, 1.0, size=(self.dim,))
                self.x = (self.x + mutation) / 2

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