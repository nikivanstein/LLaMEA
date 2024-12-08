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

        # Refine the strategy based on the best function value
        if np.random.rand() < 0.45:
            # Increase the step size for the mutation operator
            self.step_size *= 1.1
        else:
            # Decrease the step size for the mutation operator
            self.step_size /= 1.1

        return best_func

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# Description: A novel metaheuristic algorithm that combines exploration and exploitation strategies
# Code: 