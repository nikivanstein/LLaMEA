import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.pop_size = 20
        self.max_velocity = 0.1 * (5.0 - (-5.0))
        self.f = 0.5
        self.cr = 0.9
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        for _ in range(self.budget):
            w = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (self.pop_size, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity
            
            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            
            for i in range(self.pop_size):
                trial_vector = population[i]
                idx1, idx2, idx3 = np.random.choice(self.pop_size, 3, replace=False)
                if np.random.rand() < self.cr:
                    rand_dim = np.random.randint(self.dim)
                    trial_vector[rand_dim] = population[idx1, rand_dim] + self.f * (population[idx2, rand_dim] - population[idx3, rand_dim])
                
                if func(trial_vector) < pbest_fitness[i]:
                    pbest[i] = trial_vector
                    pbest_fitness[i] = func(trial_vector)
            
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
        
        return gbest