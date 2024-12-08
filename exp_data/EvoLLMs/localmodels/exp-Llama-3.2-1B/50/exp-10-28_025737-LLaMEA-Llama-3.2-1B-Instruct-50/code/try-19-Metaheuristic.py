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

        return best_func

    def mutate(self, individual):
        # Select a random individual and mutate it
        mutated_individual = individual.copy()
        mutated_individual[np.random.randint(0, self.dim)] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Select two parents and crossover them
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def __next__(self, func):
        # Generate a new individual by mutation and crossover
        new_individual = self.mutate(func(self.search_space))
        while True:
            new_individual = self.crossover(new_individual, func(self.search_space))
            new_individual_values = [func(x) for x in new_individual]
            if new_individual_values == func(new_individual_values):
                return new_individual
            else:
                new_individual = self.mutate(new_individual)