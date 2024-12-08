import numpy as np

class EnhancedDynamicPopulationSizeAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = np.full(self.pop_size, 0.1)
        self.global_mutation_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                child = parent + self.mutation_rate[i] * np.random.randn(self.dim)
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
                    self.mutation_rate[i] *= 1.02 + 0.005 * self.global_mutation_rate  # Dynamic mutation rate update
                    self.global_mutation_rate = np.mean(self.mutation_rate)  # Update global mutation rate
                    self.pop_size = int(10 * (1 - self.global_mutation_rate))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
                else:
                    self.mutation_rate[i] *= 0.98 - 0.005 * self.global_mutation_rate  # Dynamic mutation rate update
                    self.global_mutation_rate = np.mean(self.mutation_rate)  # Update global mutation rate
                    self.pop_size = int(10 * (1 - self.global_mutation_rate))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
        return population[np.argmin(fitness)]