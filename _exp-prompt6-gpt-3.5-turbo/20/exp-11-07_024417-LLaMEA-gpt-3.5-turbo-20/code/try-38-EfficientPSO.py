import numpy as np

class EfficientPSO:
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
            better_pbest = scores < pbest_scores
            pbest_scores[better_pbest] = scores[better_pbest]
            pbest_positions[better_pbest] = swarm[better_pbest].copy()
            
            gbest_index = np.argmin(pbest_scores)
            gbest_score = pbest_scores[gbest_index]
            gbest_position = pbest_positions[gbest_index].copy()

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = inertia_weight * velocities + self.c1 * r1 * (pbest_positions - swarm) + self.c2 * r2 * (gbest_position - swarm)
            swarm = np.clip(swarm + velocities, self.lower_bound, self.upper_bound)

            inertia_weight = self.inertia_max - t / self.budget * (self.inertia_max - self.inertia_min)
        
        return gbest_position