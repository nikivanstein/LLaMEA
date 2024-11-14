import numpy as np

class DynamicPopulationSizePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.max_velocity = 0.1 * (5.0 - (-5.0))
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (20, self.dim))
        velocity = np.zeros((20, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        for _ in range(self.budget):
            w = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (20, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity
            
            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]

            # Dynamic Population Size Adjustment
            if _ % 100 == 0 and _ != 0:
                new_population = np.random.uniform(-5.0, 5.0, (20, self.dim))
                new_fitness = np.array([func(ind) for ind in new_population])
                replace_indices = new_fitness < pbest_fitness
                pbest[replace_indices] = new_population[replace_indices]
                pbest_fitness[replace_indices] = new_fitness[replace_indices]
                population = new_population

        return gbest