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

        # Perform gradient-based optimization to refine the search space
        for _ in range(10):  # Limit to 10 iterations
            # Calculate the gradient of the function at the current best function value
            gradient = np.gradient(func(best_func), axis=0)

            # Select a new point in the search space with the highest gradient
            new_individual = self.search_space[np.argmax(gradient)]

            # Update the search space
            self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func