import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, bounds, mutation_prob, crossover_prob, mutation_bound, crossover_bound, budget):
        new_individual = None
        for _ in range(budget):
            if random.random() < mutation_prob:
                new_individual = self.mutation(func, bounds, mutation_bound)
            if random.random() < crossover_prob:
                new_individual = self.crossover(func, new_individual, bounds, crossover_bound)
            new_individual = func(new_individual)
            if new_individual < bounds[0]:
                new_individual = bounds[0]
            elif new_individual > bounds[1]:
                new_individual = bounds[1]
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                new_individual = random.uniform(bounds[0], bounds[1])
        return new_individual

    def mutation(self, func, bounds, mutation_bound):
        x = func(np.random.uniform(bounds[0], bounds[1]))
        while x < bounds[0] or x > bounds[1]:
            x = func(np.random.uniform(bounds[0], bounds[1]))
        if random.random() < mutation_bound:
            x = func(np.random.uniform(bounds[0], bounds[1]))
        return x

    def crossover(self, func, new_individual, bounds, crossover_bound):
        x = new_individual
        for i in range(self.dim):
            if random.random() < crossover_bound:
                x = func(np.random.uniform(bounds[0], bounds[1]))
                while x < bounds[0] or x > bounds[1]:
                    x = func(np.random.uniform(bounds[0], bounds[1]))
                if random.random() < 0.2:
                    x = func(np.random.uniform(bounds[0], bounds[1]))
        return x

# Description: Evolutionary Algorithm with Adaptive Mutation and Crossover
# Code: 