import numpy as np

class HybridPSODEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        def create_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        
        def evaluate_population(population):
            return np.array([func(individual) for individual in population])
        
        population = create_population()
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        
        for _ in range(self.max_iter):
            w = 0.5 + 0.3 * np.cos(np.pi * 2 * np.arange(self.max_iter) / self.max_iter)
            c1 = 1.5 - 1.0 * np.arange(self.max_iter) / self.max_iter
            c2 = 1.5 - 1.0 * np.arange(self.max_iter) / self.max_iter

            r1, r2 = np.random.uniform(0, 1, size=(2, self.pop_size, self.dim))
            velocity = w[:, None] * population + c1[:, None] * r1.T * (global_best - population) + c2[:, None] * r2.T * (population[best_idx] - population)
            population += velocity
            population = np.clip(population, -5.0, 5.0)
            
            idx = np.array([np.delete(np.arange(self.pop_size), i, axis=0) for i in range(self.pop_size)])
            a, b, c = population[np.random.choice(idx, (3, self.pop_size), replace=False)]
            mutant = np.clip(a + 0.8 * (b - c), -5.0, 5.0)
            
            fitness_i = func(population)
            fitness_m = func(mutant)
            
            replace_mask = fitness_m < fitness_i
            population = np.where(replace_mask[:, None], mutant.T, population).T
            fitness_i = np.where(replace_mask, fitness_m, fitness_i)
            
            best_mask = fitness_i < fitness[best_idx]
            best_idx = np.where(best_mask, np.arange(self.pop_size), best_idx)
            global_best = np.where(best_mask[:, None], population, global_best)
        
        return global_best