import numpy as np

class Improved_Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        swarm = np.random.uniform(*self.bounds, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        
        for _ in range(self.max_iter):
            r1_r2 = 2 * np.random.rand(2, self.num_particles, self.dim)
            velocity = 0.5 * velocity + r1_r2[0] * (pbest - swarm) + r1_r2[1] * (gbest - swarm)
            swarm = np.clip(swarm + velocity, *self.bounds)
            
            current_scores = np.apply_along_axis(func, 1, swarm)
            update_mask = current_scores < pbest_scores
            pbest[update_mask], pbest_scores[update_mask] = swarm[update_mask].copy(), current_scores[update_mask]
            
            gbest_idx = np.argmin(pbest_scores)
            gbest = pbest[gbest_idx].copy()

        return gbest