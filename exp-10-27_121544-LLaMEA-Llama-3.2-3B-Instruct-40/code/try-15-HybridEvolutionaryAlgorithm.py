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

        def adaptive_crossover(individual):
            new_individual = list(individual)
            for i in range(self.dim):
                if random.random() < 0.4:
                    new_individual[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
            return tuple(new_individual)

        def adaptive_mutation(individual):
            new_individual = list(individual)
            for i in range(self.dim):
                if random.random() < 0.4:
                    new_individual[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
            return tuple(new_individual)

        def evaluate_fitness(individual):
            return func(*individual)

        def optimize():
            population = [self.x0]
            for _ in range(self.budget):
                new_population = []
                while len(new_population) < self.dim:
                    parent1 = random.choice(population)
                    parent2 = random.choice(population)
                    child = (0.5 * parent1 + 0.5 * parent2)
                    child = adaptive_crossover(child)
                    child = adaptive_mutation(child)
                    new_population.append(child)
                population = new_population
            fitness = [evaluate_fitness(individual) for individual in population]
            best_individual = population[np.argmin(fitness)]
            return best_individual, np.min(fitness)

        return optimize()

# Usage
budget = 100
dim = 10
func = lambda x: x[0]**2 + x[1]**2
heacombbo = HybridEvolutionaryAlgorithm(budget, dim)
best_individual, best_fitness = heacombbo(func)
print(f"Best Individual: {best_individual}")
print(f"Best Fitness: {best_fitness}")