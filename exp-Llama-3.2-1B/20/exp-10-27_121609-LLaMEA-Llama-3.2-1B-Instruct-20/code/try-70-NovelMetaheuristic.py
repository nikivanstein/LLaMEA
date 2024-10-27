import numpy as np
import random
import copy

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = copy.deepcopy(self.population[i])
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x
                    self.population_history.append(x)

        return self.fitnesses

    def mutate(self, x):
        new_x = x.copy()
        if random.random() < 0.2:
            new_x[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return new_x

    def __str__(self):
        return f"NovelMetaheuristic(budget={self.budget}, dim={self.dim})"

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 