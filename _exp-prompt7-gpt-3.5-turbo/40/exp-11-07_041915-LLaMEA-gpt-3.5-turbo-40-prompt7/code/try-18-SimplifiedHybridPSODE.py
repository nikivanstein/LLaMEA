import numpy as np

class SimplifiedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        pop_size = 30
        swarm = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        p_best = swarm.copy()
        p_best_scores = np.array([func(ind) for ind in p_best])
        g_best = p_best[p_best_scores.argmin()]
        g_best_score = np.min(p_best_scores)

        for _ in range(self.budget // pop_size):
            r1, r2 = np.random.uniform(0, 1, (2, pop_size, self.dim))
            vel_cognitive = 1.496 * r1 * (p_best - swarm[:, np.newaxis])
            vel_social = 1.496 * r2 * (g_best - swarm)
            velocities = 0.729 * velocities + vel_cognitive.sum(axis=1) + vel_social
            swarm += velocities
            swarm = np.clip(swarm, -5.0, 5.0)
            new_scores = np.array([func(s) for s in swarm])
            improve = new_scores < p_best_scores
            p_best[improve] = swarm[improve]
            p_best_scores[improve] = new_scores[improve]
            better_global = new_scores < g_best_score
            g_best = np.where(better_global, swarm, g_best)
            g_best_score = np.where(better_global, new_scores, g_best_score)

        return g_best