import numpy as np

class ImprovedDynamicPopulationSizeAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = np.full(self.pop_size, 0.1)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                child = parent + self.mutation_rate[i] * np.random.randn(self.dim)
                child_fitness = func(child)
                if child_fitness < fitness[i]:
                    population[i] = child
                    fitness[i] = child_fitness
                    success_ratio = sum([1 for f in fitness if f < child_fitness]) / len(fitness)
                    self.mutation_rate[i] *= 1.02 if success_ratio > 0.5 else 0.98
                    self.pop_size = max(5, int(10 * success_ratio))
                    if success_ratio > 0.5:  # Increase population size for successful individuals
                        population = np.vstack((population, np.random.uniform(-5.0, 5.0, (int(5 * success_ratio), self.dim))))
            if len(population) > 2 * self.pop_size:  # Remove worst individuals if population gets too large
                worst_indices = np.argsort(fitness)[-self.pop_size:]
                population = np.delete(population, worst_indices, axis=0)
                fitness = np.delete(fitness, worst_indices)
        return population[np.argmin(fitness)]