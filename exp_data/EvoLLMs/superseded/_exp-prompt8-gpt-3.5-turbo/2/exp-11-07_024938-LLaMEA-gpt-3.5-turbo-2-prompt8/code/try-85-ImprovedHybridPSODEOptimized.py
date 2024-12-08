import numpy as np

class ImprovedHybridPSODEOptimized:
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
        gbest = np.copy(pbest[np.argmin(pbest_vals)])  # Combined random number generation here
        gbest_val = np.min(pbest_vals)
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)  # Combined random number generation here
            for i in range(self.num_particles):
                velocities[i] = self.w * velocities[i] + self.c1 * r1[i] * (pbest[i] - swarm[i]) + self.c2 * r2[i] * (gbest - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocities[i], self.bounds[0], self.bounds[1])
                
                candidate = np.clip(pbest[np.random.choice(self.num_particles)] + 0.5 * (gbest - swarm[i]), self.bounds[0], self.bounds[1])
                candidate_val = func(candidate)
                
                if candidate_val < pbest_vals[i]:
                    pbest[i] = candidate
                    pbest_vals[i] = candidate_val
                    if candidate_val < gbest_val:
                        gbest = candidate
                        gbest_val = candidate_val
        
        return gbest