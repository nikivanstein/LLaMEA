import numpy as np

class DynamicPopulationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        initial_pop_size = 20
        max_iter = self.budget // initial_pop_size
        lb, ub = -5.0 * np.ones(self.dim), 5.0 * np.ones(self.dim)

        swarm = np.random.uniform(lb, ub, (initial_pop_size, self.dim))
        velocities = np.zeros_like(swarm)
        pbest = swarm.copy()
        pbest_scores = np.array([objective_function(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        c1, c2 = 1.49445, 1.49445
        r1, r2 = np.random.rand(max_iter, initial_pop_size, self.dim), np.random.rand(max_iter, initial_pop_size, self.dim)

        for t in range(1, max_iter + 1):
            w = 0.4 + 0.4 * np.cos(0.5 * np.pi * t / max_iter)  # Adaptive inertia weight
            pop_size = initial_pop_size + 5 * np.sin(0.1 * np.pi * t)  # Dynamic population size adjustment
            velocities = w * velocities + c1 * r1[t-1] * (pbest - swarm) + c2 * r2[t-1] * (gbest - swarm)
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