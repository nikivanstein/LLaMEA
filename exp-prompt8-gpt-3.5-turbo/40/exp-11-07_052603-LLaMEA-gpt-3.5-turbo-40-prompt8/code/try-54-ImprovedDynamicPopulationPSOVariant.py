import numpy as np

class ImprovedDynamicPopulationPSOVariant:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.lb, self.ub = budget, dim, -5.0, 5.0

    def __call__(self, func):
        def obj_func(x):
            return func(x)

        pop_size, max_iter = 20, self.budget // 20
        swarm = np.random.uniform(self.lb, self.ub, (pop_size, self.dim))
        velocities, pbest = np.zeros((pop_size, self.dim)), swarm.copy()
        pbest_scores = np.array([obj_func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest, gbest_score = pbest[gbest_idx].copy(), pbest_scores[gbest_idx]
        c1, c2 = 1.49445, 1.49445
        cos_vals, sin_vals = 0.5 * np.cos(0.5 * np.pi * np.arange(1, max_iter + 1) / max_iter), 5 * np.sin(0.1 * np.pi * np.arange(1, max_iter + 1))
        r1, r2 = np.random.rand(max_iter, pop_size, self.dim), np.random.rand(max_iter, pop_size, self.dim)

        for t in range(max_iter):
            w = 0.4 + 0.4 * cos_vals[t]
            pop_size = pop_size + sin_vals[t]
            velocities = w * velocities + c1 * r1[t] * (pbest - swarm) + c2 * r2[t] * (gbest - swarm)
            swarm = np.clip(swarm + velocities, self.lb, self.ub)
            
            scores = np.array([obj_func(p) for p in swarm])
            update_idx = scores < pbest_scores
            pbest[update_idx], pbest_scores[update_idx] = swarm[update_idx], scores[update_idx]

            gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[gbest_idx] < gbest_score:
                gbest, gbest_score = pbest[gbest_idx].copy(), pbest_scores[gbest_idx]

        return gbest