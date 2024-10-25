import numpy as np

class AdaptiveMutationEvoAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop, fitness):
        idxs = np.random.choice(len(pop), 2, replace=False)
        a, b = pop[idxs[0]], pop[idxs[1]]
        mutation_scale = np.abs(np.mean(fitness) - fitness[np.argmin(fitness)])  # Adaptive mutation scale
        return x + np.random.uniform(-mutation_scale, mutation_scale) * (a - b)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget - self.pop_size):
            best_idx = np.argmin(fitness)
            new_individual = self.mutate(population[best_idx], population, fitness)
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]