import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def mutate(individual):
            if random.random() < 0.4:
                # Randomly select two indices
                i, j = random.sample(range(self.dim), 2)
                # Swap the values
                individual[i], individual[j] = individual[j], individual[i]
            return individual

        def crossover(individual1, individual2):
            # Randomly select two indices
            i, j = random.sample(range(self.dim), 2)
            # Swap the values
            individual1[i], individual2[i] = individual2[i], individual1[i]
            return individual1, individual2

        def evaluate_fitness(individual):
            return func(individual)

        def hybrid_evolution(func, bounds):
            population = [np.random.uniform(bounds[0][0], bounds[0][1], self.dim) for _ in range(100)]
            for _ in range(self.budget):
                # Evaluate the fitness
                fitness = [evaluate_fitness(individual) for individual in population]
                # Select the fittest individuals
                population = [individual for _, individual in sorted(zip(fitness, population))[:50]]
                # Perform crossover and mutation
                population = [mutate(crossover(population[0], individual)) for individual in population[1:]]
            return np.min(fitness)

        return hybrid_evolution(func, self.bounds)

# Usage:
# ```python
# import numpy as np
# from scipy.optimize import differential_evolution

# def func(x):
#     return x[0]**2 + x[1]**2

# algorithm = HybridEvolutionaryAlgorithm(100, 2)
# result = algorithm(func)
# print(result)