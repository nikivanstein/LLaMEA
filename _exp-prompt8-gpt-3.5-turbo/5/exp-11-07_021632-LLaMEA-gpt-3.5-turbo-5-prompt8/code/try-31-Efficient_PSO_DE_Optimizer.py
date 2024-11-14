import numpy as np

class Efficient_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f = budget, dim, swarm_size, p_c, f

    def __call__(self, func):
        def fitness(x):
            return func(x)

        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.array([fitness(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.swarm_size, 1), np.random.rand(self.swarm_size, 1)
            velocities = 0.5 * velocities + 2.0 * r1 * (pbest - particles) + 2.0 * r2 * (gbest - particles)
            particles += velocities

            select_idx = np.random.rand(self.swarm_size) < self.p_c
            mutants = particles[np.random.choice(self.swarm_size, (self.swarm_size, 3), replace=True)]
            v = particles + self.f * (mutants[:, 0] - mutants[:, 1] + mutants[:, 2])
            v_scores = np.array([fitness(np.clip(vi, -5.0, 5.0)) for vi in v])

            update_pbest = v_scores < pbest_scores
            pbest[update_pbest] = v[update_pbest]
            pbest_scores[update_pbest] = v_scores[update_pbest]

            update_gbest = np.min(pbest_scores) < gbest_score
            if update_gbest:
                gbest_idx = np.argmin(pbest_scores)
                gbest = pbest[gbest_idx].copy()
                gbest_score = pbest_scores[gbest_idx]

            evaluations += np.sum(select_idx)
            if evaluations >= self.budget:
                break

            particles = np.clip(particles, -5.0, 5.0)

        return gbest