import numpy as np

class EfficientHybridPSODEImproved:
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
    
    def __call__(self, func):
        swarm = self.initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = swarm
        pbest_vals = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm = np.clip(swarm + velocities, self.bounds[0], self.bounds[1])

            candidates = self.mutate(swarm[np.random.choice(self.num_particles, self.num_particles, replace=True)], gbest, swarm)
            candidate_vals = np.apply_along_axis(func, 1, candidates)
            
            updates = candidate_vals < pbest_vals
            pbest[updates] = candidates[updates]
            pbest_vals[updates] = candidate_vals[updates]
            
            improved = candidate_vals < gbest_val
            gbest = np.where(improved, candidates, gbest)
            gbest_val = np.where(improved, candidate_vals, gbest_val)
        
        return gbest

    def mutate(self, targets, best, current, f=0.5):
        return np.clip(best + f * (targets - current), self.bounds[0], self.bounds[1])