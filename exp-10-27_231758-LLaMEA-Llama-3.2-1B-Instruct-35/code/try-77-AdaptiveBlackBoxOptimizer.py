import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, alpha=0.5, beta=0.1):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.alpha = alpha
        self.beta = beta
        self.population = []

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def mutate(self, func):
        if random.random() < self.alpha:
            idx = random.randint(0, self.dim-1)
            func_values = np.copy(self.func_values)
            func_values[idx] = func_values[idx] + np.random.uniform(-self.beta, self.beta)
            self.func_values[idx] = func_values[idx]

    def crossover(self, parent1, parent2):
        if random.random() < self.beta:
            idx = random.randint(0, self.dim-1)
            parent1_values = np.copy(parent1)
            parent1_values[idx] = parent2_values[idx]
            self.func_values = np.copy(parent1_values)
        else:
            self.func_values = np.copy(parent1)

    def select(self, func1, func2):
        if random.random() < self.beta:
            self.func_values = func1
        else:
            self.func_values = func2

    def run(self, func):
        for _ in range(self.budget):
            self.select(func)
            self.mutate(func)
            self.crossover(self.func_values, func_values)
            self.func_values = func_values
            func_values = np.copy(self.func_values)
        return self.func_values

# Description: Adaptive Black Box Optimizer with adaptive mutation strategy
# Code: 