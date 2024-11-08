import numpy as np

class EnhancedDynamicPopulationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        pop_size = 20
        max_iter = self.budget // pop_size
        lb, ub = -5.0, 5.0

        swarm = np.random.uniform(lb, ub, (pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.array([objective_function(p) for p in pbest])
        
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        c1, c2 = 1.49445, 1.49445
        cos_vals = 0.5 * np.cos(0.5 * np.pi * np.arange(1, max_iter + 1) / max_iter)
        sin_vals = 5 * np.sin(0.1 * np.pi * np.arange(1, max_iter + 1))

        for t in range(max_iter):
            w = 0.4 + 0.4 * cos_vals[t]
            velocities = w * velocities + c1 * np.random.rand() * (pbest - swarm) + c2 * np.random.rand() * (gbest - swarm)
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)
            
            scores = np.array([objective_function(p) for p in swarm])
            update_idx = scores < pbest_scores
            pbest[update_idx] = swarm[update_idx]
            pbest_scores[update_idx] = scores[update_idx]

            gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[gbest_idx] < gbest_score:
                gbest = pbest[gbest_idx].copy()
                gbest_score = pbest_scores[gbest_idx]

        return gbest