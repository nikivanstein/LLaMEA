import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest_positions = swarm.copy()
        pbest_scores = np.full(self.swarm_size, np.inf)
        gbest_position = np.zeros(self.dim)
        gbest_score = np.inf
        inertia_weight = self.inertia_max

        for _ in range(self.budget):
            scores = np.array([func(p) for p in swarm])
            better_indices = np.where(scores < pbest_scores)[0]
            pbest_scores[better_indices] = scores[better_indices]
            pbest_positions[better_indices] = swarm[better_indices].copy()

            gbest_index = np.argmin(pbest_scores)
            if pbest_scores[gbest_index] < gbest_score:
                gbest_score = pbest_scores[gbest_index]
                gbest_position = swarm[gbest_index].copy()

            r = np.random.uniform(0, 1, (self.swarm_size, self.dim))
            velocities = inertia_weight * velocities + self.c1 * r * (pbest_positions - swarm) + self.c2 * r * (gbest_position - swarm)
            swarm = swarm + velocities

            swarm = np.clip(swarm, self.lower_bound, self.upper_bound)

            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
        
        return gbest_position