import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AdaptiveHyperbandBBO:
    def __init__(self, budget, dim, evolution_strategy, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_init = (-5.0, 5.0)
        self.mutation_rate = mutation_rate
        self.evolution_strategy = evolution_strategy
        self.func_evals = 0
        self.func_evals_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space_init, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        # Refine the strategy based on the fitness value
        if self.evolution_strategy == 'hyperband':
            # Hyperband strategy: increase budget by a factor of 2 when fitness value is high
            if self.func_evals_evals / self.func_evals > 0.7:
                self.budget *= 2
        elif self.evolution_strategy =='mutation':
            # Mutation strategy: randomly swap two points in the search space
            if np.random.rand() < self.mutation_rate:
                x1, x2 = np.random.choice(self.search_space_init, size=self.search_space_dim, replace=False)
                self.search_space_init = (min(self.search_space_init[0], x1), max(self.search_space_init[1], x2))
        # Store the optimized function value
        optimized_func_value = func(self.search_space)
        return optimized_func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

adaptive_hyperband = AdaptiveHyperbandBBO(budget=100, dim=10, evolution_strategy='mutation', mutation_rate=0.1)
optimized_func1 = adaptive_hyperband(test_func1)
optimized_func2 = adaptive_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Adaptive Hyperband-BBO with Evolutionary Strategy')
plt.legend()
plt.show()