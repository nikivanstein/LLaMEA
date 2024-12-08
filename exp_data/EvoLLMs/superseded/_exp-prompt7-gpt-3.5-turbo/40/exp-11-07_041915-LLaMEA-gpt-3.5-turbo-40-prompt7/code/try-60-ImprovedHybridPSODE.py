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
        g_best_idx = np.argmin(p_best_scores)
        g_best = p_best[g_best_idx].copy()
        g_best_score = p_best_scores[g_best_idx]

        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                cognitive_factor = 1.496 * r1
                social_factor = 1.496 * r2
                velocities[i] = 0.729 * velocities[i] + cognitive_factor * (p_best[i] - swarm[i]) + social_factor * (g_best - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                new_score = func(swarm[i])
                if new_score < p_best_scores[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_scores[i] = new_score
                    if new_score < g_best_score:
                        g_best = swarm[i].copy()
                        g_best_score = new_score

        return g_best