import numpy as np
import random

class AEO:
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
            idx = random.randint(0, self.dim - 1)
            self.search_space[idx] = random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.05:
            idx = random.randint(0, self.dim - 1)
            child = np.copy(parent1)
            child[idx] = parent2[idx]
            return child
        else:
            return parent1

    def __repr__(self):
        return f"AEO(budget={self.budget}, dim={self.dim})"