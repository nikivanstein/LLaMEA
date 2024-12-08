# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)

    def __call__(self, func):
        return self.metaheuristic(func)

# Example usage:
def func1(x):
    return np.sin(x)

def func2(x):
    return np.cos(x)

optimizer = BlackBoxOptimizer(100, 5)
print(optimizer(func1))  # Output: np.sin(0.0)
print(optimizer(func2))  # Output: np.cos(0.0)