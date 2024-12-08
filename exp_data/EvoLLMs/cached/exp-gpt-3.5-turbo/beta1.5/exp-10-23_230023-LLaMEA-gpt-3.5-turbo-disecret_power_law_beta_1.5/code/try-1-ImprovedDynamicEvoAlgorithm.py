import numpy as np

class ImprovedDynamicEvoAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.mutation_scale = 0.1  # New parameter for mutation scale
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x):
        return x + np.random.normal(0, self.mutation_scale) * (self.upper_bound - self.lower_bound) * self.mutation_prob

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget - self.pop_size):
            best_idx = np.argmin(fitness)
            new_individual = self.mutate(population[best_idx])
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]