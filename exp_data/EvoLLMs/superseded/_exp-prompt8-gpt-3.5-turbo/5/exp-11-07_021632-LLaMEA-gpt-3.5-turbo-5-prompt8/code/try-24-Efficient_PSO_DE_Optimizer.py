import numpy as np

class Efficient_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f = budget, dim, swarm_size, p_c, f

    def __call__(self, func):
        def fitness(x):
            return func(x)

        # Initialize particles
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

            mutation_indices = np.random.randint(self.swarm_size, size=(self.swarm_size, 3))
            mutant = particles[mutation_indices]
            v = particles + self.f * (mutant[:, 0] - mutant[:, 1] + mutant[:, 2])
            v_scores = np.array([fitness(np.clip(v_i, -5.0, 5.0)) for v_i in v])

            update_indices = np.where(v_scores < pbest_scores)
            pbest[update_indices] = v[update_indices]
            pbest_scores[update_indices] = v_scores[update_indices]

            better_indices = np.where(v_scores < gbest_score)
            gbest = np.where(better_indices, v, gbest)
            gbest_score = np.where(better_indices, v_scores, gbest_score)

            evaluations += self.swarm_size

        return gbest