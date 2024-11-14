import numpy as np

class DynamicPopulationSizePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.pop_size_min = 10
        self.pop_size_max = 30
        self.max_velocity = 0.1 * (5.0 - (-5.0))
    
    def __call__(self, func):
        w_min, w_max = 0.4, 0.9
        pop_size = self.pop_size_min
        population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        velocity = np.zeros((pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        for t in range(1, self.budget + 1):
            w = w_max - (t / self.budget) * (w_max - w_min)
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (pop_size, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity
            
            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
            
            pop_size = self.pop_size_min + ((self.pop_size_max - self.pop_size_min) * t) // self.budget
            population = np.vstack((population, np.random.uniform(-5.0, 5.0, (pop_size - population.shape[0], self.dim))))
            velocity = np.vstack((velocity, np.zeros((pop_size - velocity.shape[0], self.dim))))
        
        return gbest