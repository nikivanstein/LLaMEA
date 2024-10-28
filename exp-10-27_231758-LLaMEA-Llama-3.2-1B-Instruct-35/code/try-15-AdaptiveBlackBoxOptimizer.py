import numpy as np
from scipy.optimize import minimize

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_space = np.linspace(-5.0, 5.0, 10)
        self.population_size = 100
        self.population_deletion_probability = 0.1

    def __call__(self, func, initial_values=None):
        if initial_values is None:
            initial_values = np.zeros(self.dim)
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = initial_values
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
        return self.func_values

    def fitness(self, func_values):
        return np.mean((func_values - 4.0) ** 2)

    def __repr__(self):
        return f"AdaptiveBlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

# Description: AdaptiveBlackBoxOptimizer
# Code: 
# ```python
# ```python
# ```python
def adaptive_black_box_optimizer(budget, dim):
    optimizer = AdaptiveBlackBoxOptimizer(budget, dim)
    best_func_values = optimizer(optimizer.__call__)
    best_fitness = optimizer.fitness(best_func_values)
    return best_func_values, best_fitness

# Test the function
best_func_values, best_fitness = adaptive_black_box_optimizer(1000, 10)
print(f"Best function values: {best_func_values}")
print(f"Best fitness: {best_fitness}")

# Test the population
for _ in range(10):
    func_values = np.random.uniform(-5.0, 5.0, dim)
    best_func_values, best_fitness = adaptive_black_box_optimizer(1000, dim)
    print(f"Best function values: {best_func_values}")
    print(f"Best fitness: {best_fitness}")