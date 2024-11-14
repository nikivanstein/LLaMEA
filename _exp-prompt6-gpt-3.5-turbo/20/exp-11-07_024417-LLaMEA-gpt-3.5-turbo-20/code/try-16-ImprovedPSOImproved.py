import numpy as np

class ImprovedPSOImproved:
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

        for t in range(self.budget):
            scores = np.array([func(p) for p in swarm])

            update_pbest = scores < pbest_scores
            pbest_scores[update_pbest] = scores[update_pbest]
            pbest_positions[update_pbest] = swarm[update_pbest]

            update_gbest = pbest_scores < gbest_score
            gbest_score = np.where(update_gbest, pbest_scores, gbest_score)
            gbest_position = np.where(update_gbest[:, None], pbest_positions, gbest_position)

            r1, r2 = np.random.uniform(0, 1, (2, self.swarm_size, self.dim))
            velocities = inertia_weight * velocities + self.c1 * r1 * (pbest_positions - swarm) + self.c2 * r2 * (gbest_position - swarm)
            swarm += velocities

            np.clip(swarm, self.lower_bound, self.upper_bound, out=swarm)

            inertia_weight = self.inertia_max - (t / self.budget) * (self.inertia_max - self.inertia_min)
        
        return gbest_position