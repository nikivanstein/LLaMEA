class DynamicAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1_min = 1.49445
        self.c1_max = 2.0
        self.c2_min = 1.49445
        self.c2_max = 2.0
        self.pop_size = 20
        self.max_velocity = 0.1 * (5.0 - (-5.0))
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        for _ in range(self.budget):
            w = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            c1 = self.c1_max - (_ / self.budget) * (self.c1_max - self.c1_min)
            c2 = self.c2_max - (_ / self.budget) * (self.c2_max - self.c2_min)
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocity = w * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (np.tile(gbest, (self.pop_size, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity
            
            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
        
        return gbest