import numpy as np

class MultiParentCrossoverMetaheuristic:
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
                parents = population[np.random.choice(range(self.pop_size), 3, replace=False)]  # Selecting multiple parents
                child = np.mean(parents, axis=0) + np.mean(parents, axis=0) * 0.1 * np.random.randn(self.dim)  # Multi-parent crossover
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
                    self.mutation_rate[i] *= 1.02 + 0.005 * np.mean(self.mutation_rate)  # Dynamic mutation rate update
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
                else:
                    self.mutation_rate[i] *= 0.98 - 0.005 * np.mean(self.mutation_rate)  # Dynamic mutation rate update
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
        return population[np.argmin(fitness)]