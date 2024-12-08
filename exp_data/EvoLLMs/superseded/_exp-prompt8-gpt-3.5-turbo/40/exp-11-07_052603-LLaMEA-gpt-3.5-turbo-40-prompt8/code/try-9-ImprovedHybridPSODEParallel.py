import numpy as np
from joblib import Parallel, delayed

class ImprovedHybridPSODEParallel:
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
        pbest_scores = np.array(Parallel(n_jobs=-1)(delayed(objective_function)(p) for p in pbest))
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        w, c1, c2 = 0.5, 1.49445, 1.49445
        r1, r2 = np.random.rand(max_iter, pop_size, self.dim), np.random.rand(max_iter, pop_size, self.dim)

        for _ in range(max_iter):
            velocities = w * velocities + c1 * r1[_] * (pbest - swarm) + c2 * r2[_] * (gbest - swarm)
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)
            
            scores = np.array(Parallel(n_jobs=-1)(delayed(objective_function)(p) for p in swarm))
            update_idx = scores < pbest_scores
            pbest[update_idx] = swarm[update_idx]
            pbest_scores[update_idx] = scores[update_idx]

            gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[gbest_idx] < gbest_score:
                gbest = pbest[gbest_idx].copy()
                gbest_score = pbest_scores[gbest_idx]

        return gbest