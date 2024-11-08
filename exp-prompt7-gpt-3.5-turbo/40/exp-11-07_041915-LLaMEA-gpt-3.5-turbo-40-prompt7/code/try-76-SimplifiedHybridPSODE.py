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
            vel_cognitive = 1.496*r1*(p_best - swarm)
            vel_social = 1.496*r2*(g_best - swarm)
            velocities = 0.729*velocities + vel_cognitive + vel_social
            swarm += velocities
            swarm = np.clip(swarm, -5.0, 5.0)
            new_scores = np.array([func(p) for p in swarm])
            updates = new_scores < p_best_scores
            p_best[updates] = swarm[updates]
            p_best_scores[updates] = new_scores[updates]
            g_update = np.argmin(p_best_scores)
            if new_scores[g_update] < g_best_score:
                g_best, g_best_score = swarm[g_update], new_scores[g_update]

        return g_best