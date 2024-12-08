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
                    self.mutation_rate[i] *= 1.05 if child_fitness < np.mean(fitness) else 0.95  # Dynamic mutation rate update based on success
                    self.pop_size = int(8 + 2 * np.mean(self.mutation_rate))  # Dynamic population size adaptation for faster convergence
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))))  # Add new individuals
                else:
                    self.mutation_rate[i] *= 0.95 if child_fitness > np.mean(fitness) else 1.05  # Dynamic mutation rate adjustment
                    self.pop_size = int(8 + 2 * np.mean(self.mutation_rate))  # Dynamic population size adaptation for faster convergence
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))))  # Add new individuals
        return population[np.argmin(fitness)]