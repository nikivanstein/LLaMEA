import numpy as np

class AdaptiveMutationScaleEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.initial_mutation_scale = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop, mutation_scale):
        idxs = np.random.choice(len(pop), 2, replace=False)
        a, b = pop[idxs[0]], pop[idxs[1]]
        return x + np.random.uniform(-mutation_scale, mutation_scale) * (a - b)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        mutation_scale = self.initial_mutation_scale
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget - self.pop_size):
            best_idx = np.argmin(fitness)
            new_individual = self.mutate(population[best_idx], population, mutation_scale)
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
                mutation_scale *= 1.1  # Increase mutation scale for successful mutations
            else:
                mutation_scale *= 0.9  # Decrease mutation scale for unsuccessful mutations
        
        best_idx = np.argmin(fitness)
        return population[best_idx]