import numpy as np
import random

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.v = np.random.uniform(-1.0, 1.0, (budget, dim))
        self.f_best = np.inf
        self.x_best = None
        self.probability = 0.2

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the objective function
            f = func(self.x[i])
            
            # Update the personal best
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the global best
            if f < func(self.x_best):
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the velocity
            self.v[i] = 0.5 * (self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)))
            self.v[i] = self.v[i] + 1.0 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x_best)
            self.v[i] = self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x[i])
            
            # Update the position with probability 0.2
            if random.random() < self.probability:
                self.x[i] = self.x[i] + self.v[i]

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for _ in range(budget):
    optimizer(func)

# Evaluate the optimization results on the BBOB test suite
def evaluate_results():
    results = []
    for func in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23']:
        best_x = None
        best_f = np.inf
        for _ in range(50):
            x = np.random.uniform(-5.0, 5.0, (dim,))
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        results.append((func, best_x, best_f))
    return results

results = evaluate_results()
for func, x, f in results:
    print(f"Function: {func}, Best X: {x}, Best F: {f}")