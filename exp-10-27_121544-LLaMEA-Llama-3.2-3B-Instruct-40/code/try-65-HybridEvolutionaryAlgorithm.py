import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.probability = 0.4
        self.refine_strategy = False

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if self.refine_strategy:
            new_individual = self.refine_individual(self.x0)
        else:
            new_individual = self.x0

        if random.random() < self.probability:
            new_individual = self.mutate(new_individual)

        return self.f(new_individual, func)

    def refine_individual(self, individual):
        new_individual = list(individual)
        for i in range(self.dim):
            if random.random() < self.probability:
                new_individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        return tuple(new_individual)

    def mutate(self, individual):
        mutation_type = random.random()
        if mutation_type < 0.5:
            return self.mutate_add(individual)
        elif mutation_type < 0.8:
            return self.mutate_subtract(individual)
        else:
            return self.mutate_multiply(individual)

    def mutate_add(self, individual):
        index = random.randint(0, self.dim - 1)
        individual[index] += np.random.uniform(-1, 1)
        return tuple(individual)

    def mutate_subtract(self, individual):
        index = random.randint(0, self.dim - 1)
        individual[index] -= np.random.uniform(-1, 1)
        return tuple(individual)

    def mutate_multiply(self, individual):
        index = random.randint(0, self.dim - 1)
        factor = np.random.uniform(0.5, 1.5)
        individual[index] *= factor
        return tuple(individual)

    def f(self, individual, func):
        return func(*individual)

# Usage
budget = 100
dim = 10
algorithm = HybridEvolutionaryAlgorithm(budget, dim)

# Test the algorithm on a sample function
def f1(x):
    return x[0]**2 + x[1]**2

algorithm.refine_strategy = True
result = algorithm(f1)

print(result)