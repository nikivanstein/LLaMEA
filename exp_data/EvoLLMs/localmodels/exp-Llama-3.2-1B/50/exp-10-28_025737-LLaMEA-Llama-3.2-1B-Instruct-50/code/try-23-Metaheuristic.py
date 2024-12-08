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
        if len(self.search_space) < 10:
            self.search_space = self.search_space + [x for x in self.search_space if x not in best_func]
        elif random.random() < 0.45:
            self.search_space = [x for x in self.search_space if x not in best_func]
        else:
            self.search_space = [x for x in self.search_space if x in best_func]

        return best_func

# Example usage
def func(x):
    return np.sum(x)

algorithm = Metaheuristic(100, 5)
best_func = algorithm(func)

# Print the result
print("Best function:", best_func)
print("Best fitness:", func(best_func))