import numpy as np

class Improved_Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f = budget, dim, swarm_size, p_c, f

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.swarm_size), np.random.rand(self.swarm_size)
            pbest_diff = pbest - particles
            gbest_diff = np.tile(gbest, (self.swarm_size, 1)) - particles

            particles += 0.5 * pbest_diff + 2.0 * r1.reshape(-1, 1) * pbest_diff + 2.0 * r2.reshape(-1, 1) * gbest_diff

            mutation_indices = np.random.randint(0, self.swarm_size, (self.swarm_size, 3))
            mutant = particles[mutation_indices]
            v = particles + self.f * (mutant[:, 0] - mutant[:, 1] + mutant[:, 2])
            v = np.clip(v, -5.0, 5.0)
            v_scores = np.apply_along_axis(func, 1, v)

            improve_pbest_mask = v_scores < pbest_scores
            pbest[improve_pbest_mask], pbest_scores[improve_pbest_mask] = v[improve_pbest_mask], v_scores[improve_pbest_mask]

            new_gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[new_gbest_idx] < gbest_score:
                gbest, gbest_score = pbest[new_gbest_idx].copy(), pbest_scores[new_gbest_idx]

            evaluations += self.swarm_size

        return gbest