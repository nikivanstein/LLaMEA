import numpy as np

class DynamicImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        pop_size = 20
        max_iter = self.budget // pop_size
        lb, ub = -5.0 * np.ones(self.dim), 5.0 * np.ones(self.dim)

        swarm = np.random.uniform(lb, ub, (pop_size, self.dim))
        velocities = np.zeros_like(swarm)
        pbest = swarm.copy()
        pbest_scores = np.array([objective_function(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        w_min, w_max = 0.4, 0.9
        c_min, c_max = 1.2, 2.2

        r1, r2 = np.random.rand(max_iter, pop_size, self.dim), np.random.rand(max_iter, pop_size, self.dim)

        for t in range(max_iter):
            w = w_max - (w_max - w_min) * t / max_iter
            c1 = c_max - (c_max - c_min) * t / max_iter
            c2 = c_max - (c_max - c_min) * t / max_iter

            velocities = w * velocities + c1 * r1[t] * (pbest - swarm) + c2 * r2[t] * (gbest - swarm)
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