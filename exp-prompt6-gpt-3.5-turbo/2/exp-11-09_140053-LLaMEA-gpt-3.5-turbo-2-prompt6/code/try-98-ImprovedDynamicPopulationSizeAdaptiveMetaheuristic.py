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
                    improvement_rate = (fitness[i] - child_fitness) / fitness[i]
                    adjustment = 0.1 * improvement_rate  # Dynamic mutation rate adjustment
                    self.mutation_rate[i] *= 1 + adjustment
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))
                else:
                    self.mutation_rate[i] *= 0.95  # Reduced mutation rate for exploitation
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))
        return population[np.argmin(fitness)]