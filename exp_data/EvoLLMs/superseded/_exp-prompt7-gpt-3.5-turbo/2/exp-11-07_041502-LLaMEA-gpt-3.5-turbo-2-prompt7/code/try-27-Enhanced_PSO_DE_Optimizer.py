import numpy as np

class Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.bounds = (-5.0, 5.0)

    def optimize_swarm(self, swarm, velocity, pbest, pbest_scores, gbest, gbest_idx, func):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(2)
            velocity[i] = 0.5 * velocity[i] + 2 * r1 * (pbest[i] - swarm[i]) + 2 * r2 * (gbest - swarm[i])
            swarm[i] = np.clip(swarm[i] + velocity[i], *self.bounds)
            
            current_score = func(swarm[i])
            if current_score < pbest_scores[i]:
                pbest[i], pbest_scores[i] = swarm[i].copy(), current_score

            if current_score < pbest_scores[gbest_idx]:
                gbest, gbest_idx = pbest[i].copy(), i
        
        return gbest, gbest_idx
    
    def __call__(self, func):
        swarm = np.random.uniform(*self.bounds, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        
        for _ in range(self.max_iter):
            gbest, gbest_idx = self.optimize_swarm(swarm, velocity, pbest, pbest_scores, gbest, gbest_idx, func)

        return gbest