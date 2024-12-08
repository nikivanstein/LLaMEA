import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        swarm = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        p_best = swarm.copy()
        p_best_scores = np.array([func(ind) for ind in p_best])
        g_best_index = np.argmin(p_best_scores)
        g_best = p_best[g_best_index]
        g_best_score = p_best_scores[g_best_index]

        for _ in range(self.budget // pop_size):
            r1, r2 = np.random.uniform(0, 1, (2, pop_size, self.dim))
            vel_cognitive = 1.496 * r1 * (p_best - swarm)
            vel_social = 1.496 * r2 * (g_best - swarm)
            velocities = 0.729 * velocities + vel_cognitive + vel_social
            swarm += velocities
            swarm = np.clip(swarm, -5.0, 5.0)
            new_scores = np.array([func(s) for s in swarm])
            update_mask = new_scores < p_best_scores
            p_best[update_mask] = swarm[update_mask]
            p_best_scores[update_mask] = new_scores[update_mask]
            g_best_index = np.argmin(p_best_scores)
            if p_best_scores[g_best_index] < g_best_score:
                g_best = p_best[g_best_index]
                g_best_score = p_best_scores[g_best_index]

        return g_best