import numpy as np
from scipy.optimize import minimize
from random import randint

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        """Randomly mutate an individual to refine its strategy"""
        if randint(0, self.budget) < 0.05:
            # Change the individual's value to a new random value
            new_individual = np.random.rand(self.dim)
            new_individual = (new_individual - self.search_space) / 2
            return new_individual
        else:
            return individual

    def crossover(self, parent1, parent2):
        """Perform crossover to combine two parents to create a new child"""
        if randint(0, self.budget) < 0.05:
            # Select a random point to crossover at
            crossover_point = np.random.randint(0, self.dim)
            # Combine the two parents
            child = (parent1[:crossover_point] + parent2[crossover_point:])
            return child
        else:
            return parent1

# Description: Hebbian Evolutionary Bias
# Code: 