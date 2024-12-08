import numpy as np

class HybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 2.0
        self.c2 = 2.0
        self.mutation_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.max_iter):
            best_idx = np.argmin(fitness)
            global_best = population[best_idx].copy()
            
            for i in range(self.population_size):
                # PSO update
                velocity = np.random.rand(self.dim) * velocity + self.c1 * np.random.rand(self.dim) * (
                        population[best_idx] - population[i]) + self.c2 * np.random.rand(self.dim) * (
                                   global_best - population[i])
                population[i] += velocity
                
                # GA mutation
                if np.random.rand() < self.mutation_rate:
                    population[i] += np.random.uniform(-0.1, 0.1, self.dim)
                
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]