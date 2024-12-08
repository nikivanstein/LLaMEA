import random
import numpy as np
import operator

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

        # Select the next individual based on the probability of refinement
        next_individual = self.search_space[np.random.choice(len(self.search_space), p=[0.45]*len(self.search_space))]
        return next_individual

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize the population with the selected solution
        population = [self.search_space[np.random.choice(len(self.search_space), p=[0.45]*len(self.search_space))]]

        # Evaluate the function a limited number of times
        for _ in range(self.budget):
            new_individual = self.search_space[np.random.choice(len(self.search_space), p=[0.45]*len(self.search_space))]

            # Select the best function value
            best_func = max(set([func(x) for x in self.search_space]), key=func)

            # Update the search space
            population.append(new_individual)
            self.search_space = [x for x in self.search_space if x not in best_func]

        return population

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 