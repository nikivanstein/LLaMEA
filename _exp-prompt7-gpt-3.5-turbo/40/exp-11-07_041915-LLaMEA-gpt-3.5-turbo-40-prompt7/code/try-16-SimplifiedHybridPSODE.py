import numpy as np

class SimplifiedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size = budget, dim, 30
        self.swarm = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.p_best = self.swarm.copy()
        self.p_best_scores = np.array([func(ind) for ind in self.p_best])
        self.g_best = self.p_best[self.p_best_scores.argmin()]
        self.g_best_score = np.min(self.p_best_scores)

    def __call__(self, func):
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                vel_cognitive = 1.496 * r1 * (self.p_best[i] - self.swarm[i])
                vel_social = 1.496 * r2 * (self.g_best - self.swarm[i])
                self.velocities[i] = 0.729 * self.velocities[i] + vel_cognitive + vel_social
                self.swarm[i] += self.velocities[i]
                self.swarm[i] = np.clip(self.swarm[i], -5.0, 5.0)
                new_score = func(self.swarm[i])
                if new_score < self.p_best_scores[i]:
                    self.p_best[i], self.p_best_scores[i] = self.swarm[i].copy(), new_score
                    if new_score < self.g_best_score:
                        self.g_best, self.g_best_score = self.swarm[i].copy(), new_score

        return self.g_best