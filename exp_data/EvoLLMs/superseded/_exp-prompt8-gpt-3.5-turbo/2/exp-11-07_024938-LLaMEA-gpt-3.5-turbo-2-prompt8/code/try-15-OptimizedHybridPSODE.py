import numpy as np

class OptimizedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.bounds = (-5.0, 5.0)
    
    def initialize_particles(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
    
    def mutate(self, target, best, current, f=0.5):
        return np.clip(best + f * (target - current), self.bounds[0], self.bounds[1])
    
    def __call__(self, func):
        swarm = self.initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_vals = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_vals[gbest_idx]
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_i = swarm[i]
                velocities_i = velocities[i]
                pbest_i = pbest[i]
                pbest_val_i = pbest_vals[i]
                
                velocities_i = self.w * velocities_i + self.c1 * r1 * (pbest_i - swarm_i) + self.c2 * r2 * (gbest - swarm_i)
                swarm_i = np.clip(swarm_i + velocities_i, self.bounds[0], self.bounds[1])
                
                candidate = self.mutate(swarm[np.random.choice(self.num_particles)], gbest, swarm_i)
                candidate_val = func(candidate)
                
                if candidate_val < pbest_val_i:
                    pbest[i], pbest_vals[i] = candidate, candidate_val
                    if candidate_val < gbest_val:
                        gbest, gbest_val = candidate, candidate_val
        
        return gbest