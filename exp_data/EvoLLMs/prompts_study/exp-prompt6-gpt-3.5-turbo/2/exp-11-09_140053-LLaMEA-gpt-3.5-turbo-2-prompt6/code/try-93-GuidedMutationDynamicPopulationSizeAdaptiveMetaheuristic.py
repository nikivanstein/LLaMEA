import numpy as np

class GuidedMutationDynamicPopulationSizeAdaptiveMetaheuristic:
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
                mutation_direction = np.sign(np.random.randn(self.dim))  # Guided mutation by random direction bias
                child = parent + self.mutation_rate[i] * mutation_direction
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
                    self.mutation_rate[i] *= 1.02 + 0.005 * np.mean(self.mutation_rate)  
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  
                else:
                    self.mutation_rate[i] *= 0.98 - 0.005 * np.mean(self.mutation_rate)  
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  
        return population[np.argmin(fitness)]