import numpy as np
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def mutate(individual):
            if np.random.rand() < 0.4:
                individual = individual + np.random.uniform(-1, 1, self.dim)
            return individual

        def crossover(parent1, parent2):
            if np.random.rand() < 0.5:
                child = parent1 + parent2 - np.sum(parent1 + parent2, axis=0) / 2
            else:
                child = parent1 - parent2
            return child

        def hybrid_evolution(func):
            population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(100)]
            for _ in range(self.budget):
                new_population = []
                for individual in population:
                    mutated_individual = mutate(individual)
                    new_population.append(mutated_individual)
                population = new_population
                new_population = []
                for i in range(len(population)):
                    parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
                    child = crossover(parent1, parent2)
                    new_population.append(child)
                population = new_population
            return differential_evolution(func, self.bounds, x0=np.mean(population, axis=0))

        return hybrid_evolution(func)

# Usage
budget = 100
dim = 5
func = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2
heacombbo = HEACOMBBO(budget, dim)
result = heacombbo(func)
print(result)