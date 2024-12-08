import numpy as np

class CooperativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 20
        self.swarms = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.personal_best = self.swarms.copy()
        self.global_best = self.swarms[np.argmin([func(p) for p in self.swarms])]

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = 0.5 * self.velocities[i] + 2.0 * r1 * (self.personal_best[i] - self.swarms[i]) + 2.0 * r2 * (self.global_best - self.swarms[i])
                self.swarms[i] = np.clip(self.swarms[i] + self.velocities[i], self.lower_bound, self.upper_bound)
                if func(self.swarms[i]) < func(self.personal_best[i]):
                    self.personal_best[i] = self.swarms[i]
                    if func(self.swarms[i]) < func(self.global_best):
                        self.global_best = self.swarms[i]
        return self.global_best