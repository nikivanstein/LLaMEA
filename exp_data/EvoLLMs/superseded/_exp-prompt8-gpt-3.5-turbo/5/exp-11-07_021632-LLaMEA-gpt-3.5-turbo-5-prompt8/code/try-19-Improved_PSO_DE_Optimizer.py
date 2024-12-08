import numpy as np

class Improved_PSO_DE_Optimizer:
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

            update_mask = np.random.rand(self.swarm_size) < self.p_c
            to_mutate = np.where(update_mask)[0]
            mutants = particles[np.random.choice(self.swarm_size, (len(to_mutate), 3), replace=False)]
            v = particles[to_mutate] + self.f * (mutants[:, 0] - mutants[:, 1] + mutants[:, 2])
            v_scores = np.array([fitness(np.clip(val, -5.0, 5.0)) for val in v])

            improve_mask = v_scores < pbest_scores[to_mutate]
            pbest[to_mutate[improve_mask]] = v[improve_mask]
            pbest_scores[to_mutate[improve_mask]] = v_scores[improve_mask]

            better_gbest_mask = pbest_scores < gbest_score
            gbest = pbest[better_gbest_mask][0].copy()
            gbest_score = pbest_scores[better_gbest_mask][0]

            evaluations += len(to_mutate)

        return gbest