import numpy as np
import random
from scipy.optimize import minimize

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
        if random.random() < 0.05:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return individual

    def crossover(self, parent1, parent2):
        child = np.concatenate((parent1[:self.search_space.shape[0] // 2], parent2[self.search_space.shape[0] // 2:]))
        return child

    def selection(self, individuals):
        return np.random.choice(len(individuals), size=self.budget, replace=False)

# One-line description: Evolutionary Algorithm for Black Box Optimization
# Code: 