import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        swarm = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        p_best = swarm.copy()
        p_best_scores = np.array([func(ind) for ind in p_best])
        g_best = p_best[p_best_scores.argmin()]
        g_best_score = np.min(p_best_scores)

        for _ in range(self.budget // pop_size):
            r1, r2 = np.random.uniform(0, 1, (2, pop_size))
            vel_cognitive = 1.496*r1*(p_best - swarm).T
            vel_social = 1.496*r2*(g_best - swarm).T
            velocities = 0.729*velocities + vel_cognitive.T + vel_social.T
            swarm += velocities
            swarm = np.clip(swarm, -5.0, 5.0)
            new_scores = np.array([func(s) for s in swarm])
            update_indices = new_scores < p_best_scores
            p_best[update_indices] = swarm[update_indices]
            p_best_scores[update_indices] = new_scores[update_indices]
            update_gbest = p_best_scores < g_best_score
            g_best = np.where(update_gbest, p_best, g_best)
            g_best_score = np.where(update_gbest, p_best_scores, g_best_score)

        return g_best