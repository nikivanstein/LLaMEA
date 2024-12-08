import numpy as np
import random

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

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        # Refine the strategy using probability 0.45
        new_individual = self.search_space[:self.dim // 2] + [best_func]
        new_individual = self.evaluate_fitness(new_individual)
        if np.random.rand() < 0.45:
            new_individual = self.search_space[self.dim // 2:] + [best_func]

        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the function at the new individual
        func_values = [func(x) for x in individual]
        best_func = max(set(func_values), key=func_values.count)
        return best_func

# Initialize the selected solution
selected_solution = NovelMetaheuristicAlgorithm(1000, 10)