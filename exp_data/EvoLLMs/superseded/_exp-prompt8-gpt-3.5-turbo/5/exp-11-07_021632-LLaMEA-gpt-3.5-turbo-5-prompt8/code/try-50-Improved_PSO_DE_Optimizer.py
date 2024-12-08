import numpy as np

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5, batch_size=5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f, self.batch_size = budget, dim, swarm_size, p_c, f, batch_size

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(0, self.swarm_size, self.batch_size):
                batch_indices = range(i, min(i + self.batch_size, self.swarm_size))
                batch_particles = particles[batch_indices]
                batch_pbest = pbest[batch_indices]
                
                r1, r2 = np.random.rand(len(batch_indices)), np.random.rand(len(batch_indices))
                pbest_diff = batch_particles - batch_pbest
                gbest_diff = batch_particles - gbest
                
                particles[batch_indices] += 0.5 * pbest_diff + 2.0 * r1[:, None] * pbest_diff + 2.0 * r2[:, None] * gbest_diff

                mutate_indices = np.random.choice(self.swarm_size, (len(batch_indices), 3), replace=False)
                mutants = particles[mutate_indices]
                v = batch_particles + self.f * (mutants[:, 0] - mutants[:, 1] + mutants[:, 2])
                v_scores = np.apply_along_axis(func, 1, np.clip(v, -5.0, 5.0))

                improved_indices = np.where(v_scores < pbest_scores[batch_indices])[0]
                pbest[batch_indices[improved_indices]] = v[improved_indices]
                pbest_scores[batch_indices[improved_indices]] = v_scores[improved_indices]

                improved_gbest = np.argmin(pbest_scores)
                if pbest_scores[improved_gbest] < gbest_score:
                    gbest, gbest_score = pbest[improved_gbest].copy(), pbest_scores[improved_gbest]

                evaluations += len(batch_indices)
                if evaluations >= self.budget:
                    break

        return gbest