import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None
        self.prob_local = 0.2857142857142857

    def __call__(self, func):
        for _ in range(self.budget):
            # Global Exploration using Differential Evolution
            f_x = func(self.x)
            if f_x < self.f_best:
                self.f_best = f_x
                self.x_best = self.x.copy()
            
            # Local Exploration using Random Search
            if np.random.rand() < self.prob_local:
                # Generate a random direction
                u = np.random.uniform(-1.0, 1.0, size=(self.dim,))
                u = u / np.linalg.norm(u)
                # Move in the random direction
                x_p = self.x + u * np.random.uniform(0.1, 1.0, size=(self.dim))
                # Evaluate the function at the new point
                f_x_p = func(x_p)
                # Update the best point if necessary
                if f_x_p < self.f_best:
                    self.f_best = f_x_p
                    self.x_best = x_p.copy()

            # Update the current position using Differential Evolution
            if np.random.rand() < 0.2:
                # Generate two random points
                x1 = self.x.copy()
                x2 = self.x.copy()
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        x1[i] = self.x[i] + np.random.uniform(-1.0, 1.0)
                    else:
                        x1[i] = self.x[i] - np.random.uniform(-1.0, 1.0)
                    if np.random.rand() < 0.5:
                        x2[i] = self.x[i] + np.random.uniform(-1.0, 1.0)
                    else:
                        x2[i] = self.x[i] - np.random.uniform(-1.0, 1.0)
                # Calculate the difference between the two points
                f_x1 = func(x1)
                f_x2 = func(x2)
                # Update the current position using the better point
                if f_x1 < f_x2:
                    self.x = x1
                else:
                    self.x = x2

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