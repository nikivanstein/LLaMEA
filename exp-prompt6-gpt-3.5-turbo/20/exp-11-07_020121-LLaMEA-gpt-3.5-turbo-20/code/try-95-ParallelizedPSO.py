import numpy as np

class ParallelizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.496
        self.c2 = 1.496
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.velocities = np.zeros((budget, dim))

    def __call__(self, func):
        fitness_values = np.array([func(x) for x in self.population])
        g_best_idx = np.argmin(fitness_values)
        g_best = self.population[g_best_idx]
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.c1 * r1 * (self.population - self.population) 
            social_component = self.c2 * r2 * (g_best - self.population)
            
            self.velocities = self.w * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, -5.0, 5.0)
            
            new_fitness_values = np.array([func(x) for x in self.population])
            new_g_best_idx = np.argmin(new_fitness_values)
            update_indices = new_fitness_values < fitness_values
            
            g_best_idx = np.where(update_indices, new_g_best_idx, g_best_idx)
            g_best = np.where(update_indices, self.population[new_g_best_idx], g_best)
            fitness_values = np.where(update_indices, new_fitness_values, fitness_values)
        
        return g_best