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

        # Refine the strategy
        if random.random() < 0.45:
            # Add a new dimension to the search space
            new_dim = np.random.uniform(-5.0, 5.0)
            new_search_space = self.search_space + [new_dim]
            self.search_space = new_search_space

        return best_func

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 