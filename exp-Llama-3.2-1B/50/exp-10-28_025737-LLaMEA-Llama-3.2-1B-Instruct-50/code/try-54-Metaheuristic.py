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

class NovelMetaheuristicAlgorithm:
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

        # Refine the strategy based on the best function value
        if random.random() < 0.45:
            # Explore the search space by swapping two random elements
            i, j = random.sample(range(self.dim), 2)
            self.search_space[i], self.search_space[j] = self.search_space[j], self.search_space[i]
            self.search_space[i] = random.uniform(-5.0, 5.0)
            self.search_space[j] = random.uniform(-5.0, 5.0)

        return best_func

# Example usage:
algorithm = NovelMetaheuristicAlgorithm(100, 10)
best_func = algorithm(__call__, np.sin)

# Print the results
print("Best function:", best_func)
print("Fitness:", np.sin(best_func))

# Evaluate the function a limited number of times
num_evals = min(algorithm.budget, len(np.sin))
func_values = [np.sin(x) for x in random.sample(range(0, 100, 0.01), num_evals)]

# Select the best function value
best_func = max(set(func_values), key=func_values.count)

# Print the results
print("Best function:", best_func)
print("Fitness:", np.sin(best_func))