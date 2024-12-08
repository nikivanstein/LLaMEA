import numpy as np

class ImprovedHybridPSODEEnhanced:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.lb, self.ub = -5.0 * np.ones(dim), 5.0 * np.ones(dim)
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.swarm = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.velocities = np.zeros_like(self.swarm)
        self.pbest = self.swarm.copy()
        self.pbest_scores = np.array([func(x) for x in self.pbest])
        self.gbest_idx = np.argmin(self.pbest_scores)
        self.gbest = self.pbest[self.gbest_idx].copy()
        self.gbest_score = self.pbest_scores[self.gbest_idx]
        self.w, self.c1, self.c2 = 0.5, 1.49445, 1.49445
        self.r1, self.r2 = np.random.rand(self.max_iter, self.pop_size, dim), np.random.rand(self.max_iter, self.pop_size, dim)

        for _ in range(self.max_iter):
            self.velocities = self.w * self.velocities + self.c1 * self.r1[_] * (self.pbest - self.swarm) + self.c2 * self.r2[_] * (self.gbest - self.swarm)
            self.swarm += self.velocities
            self.swarm = np.clip(self.swarm, self.lb, self.ub)
            scores = np.array([func(x) for x in self.swarm])
            update_idx = scores < self.pbest_scores
            self.pbest[update_idx] = self.swarm[update_idx]
            self.pbest_scores[update_idx] = scores[update_idx]
            self.gbest_idx = np.argmin(self.pbest_scores)
            if self.pbest_scores[self.gbest_idx] < self.gbest_score:
                self.gbest = self.pbest[self.gbest_idx].copy()
                self.gbest_score = self.pbest_scores[self.gbest_idx]

        return self.gbest