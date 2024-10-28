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
        new_individual = self.evaluate_fitness(best_func)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func, new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        fitness = 0
        for dim, value in zip(self.dim, individual):
            fitness += np.abs(value - 5.0)
        return fitness

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 