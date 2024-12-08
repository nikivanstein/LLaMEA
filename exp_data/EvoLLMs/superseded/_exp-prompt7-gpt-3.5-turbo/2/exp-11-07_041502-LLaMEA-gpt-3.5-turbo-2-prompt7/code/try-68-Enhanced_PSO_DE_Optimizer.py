import numpy as np

class Enhanced_PSO_DE_Optimizer:
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
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)
            velocity = 0.5 * velocity + 2 * r1 * (pbest - swarm) + 2 * r2 * (gbest - swarm)
            swarm = np.clip(swarm + velocity, *self.bounds)
            
            current_scores = np.apply_along_axis(func, 1, swarm)
            replace_mask = current_scores < pbest_scores
            pbest[replace_mask] = swarm[replace_mask]
            pbest_scores[replace_mask] = current_scores[replace_mask]

            new_gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[new_gbest_idx] < pbest_scores[gbest_idx]:
                gbest, gbest_idx = pbest[new_gbest_idx].copy(), new_gbest_idx

        return gbest