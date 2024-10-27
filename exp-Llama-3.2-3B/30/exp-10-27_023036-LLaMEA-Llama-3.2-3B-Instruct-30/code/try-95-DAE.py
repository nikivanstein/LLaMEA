import numpy as np
import random

class DAE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.mutate_prob = 0.3
        self.adapt_prob = 0.7
        self.expl_prob = 0.1

    def __call__(self, func):
        if len(self.population) == 0:
            self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(20)]

        for _ in range(self.budget):
            # Select best 5 individuals
            self.population.sort(key=lambda x: func(x))
            self.population = self.population[:5]

            # Adapt individuals
            for i in range(len(self.population)):
                if random.random() < self.adapt_prob:
                    self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)

            # Explore individuals
            for i in range(len(self.population)):
                if random.random() < self.expl_prob:
                    self.population[i] += np.random.uniform(-0.5, 0.5, self.dim)

            # Mutate individuals
            for i in range(len(self.population)):
                if random.random() < self.mutate_prob:
                    idx = random.randint(0, self.dim-1)
                    self.population[i][idx] += np.random.uniform(-0.5, 0.5)

            # Evaluate function
            for individual in self.population:
                func(individual)

# Example usage
def func(x):
    return np.sum(x**2)

dae = DAE(100, 2)
dae(func)