import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None
        self.pso_alpha = 0.861
        self.pso_beta = 0.333
        self.pso_gamma = 2.05

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

            # Update the current position using PSO
            r1 = np.random.uniform(0, 1, size=self.dim)
            r2 = np.random.uniform(0, 1, size=self.dim)
            x_pso = self.x.copy()
            for i in range(self.dim):
                if r1[i] < 0.3:
                    x_pso[i] = self.x[i] + self.pso_alpha * (self.x_best[i] - self.x[i])
                elif r1[i] < 0.6:
                    x_pso[i] = self.x[i] + self.pso_beta * (self.x[i] - self.x_best[i])
                else:
                    x_pso[i] = self.x[i] - self.pso_gamma * (self.x[i] - self.x_best[i])
                if r2[i] < 0.5:
                    x_pso[i] = x_pso[i] + np.random.uniform(-1.0, 1.0)
                else:
                    x_pso[i] = x_pso[i] - np.random.uniform(-1.0, 1.0)
            f_x_pso = func(x_pso)
            if f_x_pso < self.f_best:
                self.f_best = f_x_pso
                self.x_best = x_pso.copy()

            # Crossover
            if np.random.rand() < 0.061224489795918366:
                x_cross = np.zeros(self.dim)
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        x_cross[i] = self.x[i]
                    else:
                        x_cross[i] = x_pso[i]
                f_x_cross = func(x_cross)
                if f_x_cross < self.f_best:
                    self.f_best = f_x_cross
                    self.x_best = x_cross.copy()

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