import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.crossover_probability = 0.4
        self.mutation_probability = 0.1
        self.adaptive_strategy = True

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(100)]

        for _ in range(self.budget):
            fitness = [func(individual) for individual in population]
            best_individual = np.argmin(fitness)
            best_individual = population[best_individual]

            if random.random() < self.crossover_probability:
                parent1, parent2 = random.sample(population, 2)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                population.append(child1)
                population.append(child2)

            if random.random() < self.mutation_probability:
                if self.adaptive_strategy:
                    if random.random() < 0.5:
                        mutation = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
                    else:
                        mutation = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
                else:
                    mutation = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
                population.append(best_individual + mutation)

            population.sort(key=lambda individual: func(individual))

        return np.min([func(individual) for individual in population]), np.min([func(individual) for individual in population])

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.dim):
            if random.random() < self.crossover_probability:
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            else:
                child.append(random.choice([parent1[i], parent2[i]]))
        return child

# Usage
func = lambda x: x[0]**2 + x[1]**2
algorithm = HybridEvolutionaryAlgorithm(100, 2)
result = algorithm(func)
print(result)