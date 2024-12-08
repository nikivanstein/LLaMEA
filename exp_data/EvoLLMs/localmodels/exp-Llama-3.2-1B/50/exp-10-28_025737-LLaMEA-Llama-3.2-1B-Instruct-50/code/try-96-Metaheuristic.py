# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

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

        # Refine the strategy based on the budget
        if len(self.search_space) > 0:
            # Evaluate the function with a reduced search space
            num_evals = min(self.budget, len(func(self.search_space[::2])))
            func_values = [func(x) for x in random.sample(self.search_space[::2], num_evals)]

            # Select the best function value with the refined strategy
            best_func = max(set(func_values), key=func_values.count)

            # Update the search space with the best function value
            self.search_space = [x for x in self.search_space if x in best_func]

        return best_func

class BBOB:
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

        # Refine the strategy based on the budget
        if len(self.search_space) > 0:
            # Evaluate the function with a reduced search space
            num_evals = min(self.budget, len(func(self.search_space[::2])))
            func_values = [func(x) for x in random.sample(self.search_space[::2], num_evals)]

            # Select the best function value with the refined strategy
            best_func = max(set(func_values), key=func_values.count)

            # Update the search space with the best function value
            self.search_space = [x for x in self.search_space if x in best_func]

        return best_func

# Initialize the metaheuristic algorithm
metaheuristic = Metaheuristic(100, 10)
# Initialize the BBOB test suite
bboo = BBOB(100, 10)

# Evaluate the function 100 times
for _ in range(100):
    func = bboo.__call__(metaheuristic)
    best_func = metaheuristic(func)
    print(f"Best function: {best_func}, Score: {best_func.__call__(bboo.func)}")