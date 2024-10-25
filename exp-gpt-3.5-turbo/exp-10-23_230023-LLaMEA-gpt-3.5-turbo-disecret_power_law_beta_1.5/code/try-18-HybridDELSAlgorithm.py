import numpy as np

class HybridDELSAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.mutation_scale = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop):
        idxs = np.random.choice(len(pop), 2, replace=False)
        a, b = pop[idxs[0]], pop[idxs[1]]
        return x + np.random.uniform(-self.mutation_scale, self.mutation_scale) * (a - b)

    def local_search(self, x, func):
        new_x = x + np.random.uniform(-0.1, 0.1, size=self.dim)
        if func(new_x) < func(x):
            return new_x
        return x

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget - self.pop_size):
            best_idx = np.argmin(fitness)
            new_individual = self.mutate(population[best_idx], population)
            new_individual = self.local_search(new_individual, func)
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]