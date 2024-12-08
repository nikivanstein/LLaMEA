import numpy as np

class NovelAdaptiveDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.adaptive_mutation_scale = 0.2
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop):
        idxs = np.random.choice(len(pop), 3, replace=False)
        a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
        return np.clip(a + self.adaptive_mutation_scale * (b - c), self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget - self.pop_size):
            best_idx = np.argmin(fitness)
            new_individual = self.mutate(population[best_idx], population)
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]