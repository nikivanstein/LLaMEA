import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget, self.dim, pop_size = budget, dim, 30
        self.swarm, self.velocities = np.random.uniform(-5.0, 5.0, (pop_size, dim)), np.zeros((pop_size, dim))
        self.p_best, self.p_best_scores = self.swarm.copy(), np.array([func(ind) for ind in self.swarm])
        self.g_best, self.g_best_score = self.p_best[self.p_best_scores.argmin()], np.min(self.p_best_scores)

    def __call__(self, func):
        for _ in range(self.budget // len(self.swarm)):
            for i in range(len(self.swarm)):
                r1, r2 = np.random.uniform(0, 1, 2)
                cognitive, social = 1.496 * r1 * (self.p_best[i] - self.swarm[i]), 1.496 * r2 * (self.g_best - self.swarm[i])
                self.velocities[i] = 0.729 * self.velocities[i] + cognitive + social
                self.swarm[i] = np.clip(self.swarm[i] + self.velocities[i], -5.0, 5.0)
                new_score = func(self.swarm[i])
                if new_score < self.p_best_scores[i]:
                    self.p_best[i], self.p_best_scores[i] = self.swarm[i].copy(), new_score
                    if new_score < self.g_best_score:
                        self.g_best, self.g_best_score = self.swarm[i].copy(), new_score
        return self.g_best