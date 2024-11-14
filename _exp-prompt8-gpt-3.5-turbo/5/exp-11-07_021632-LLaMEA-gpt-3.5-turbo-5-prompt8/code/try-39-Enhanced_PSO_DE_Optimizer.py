import numpy as np

class Enhanced_PSO_DE_Optimizer:
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

            update_indices = np.random.rand(self.swarm_size) < self.p_c
            selected_particles = particles[update_indices]
            selected_pbest = pbest[update_indices]
            pbest_scores_selected = pbest_scores[update_indices]

            mutants = particles[np.random.choice(self.swarm_size, (np.sum(update_indices), 3), replace=False)]
            v = selected_particles + self.f * (mutants[:, 0] - mutants[:, 1] + mutants[:, 2])
            v_scores = np.array([fitness(np.clip(v_x, -5.0, 5.0)) for v_x in v])

            improved_indices = v_scores < pbest_scores_selected
            pbest[update_indices][improved_indices] = v[improved_indices]
            pbest_scores[update_indices][improved_indices] = v_scores[improved_indices]

            improved_global = np.min(pbest_scores) < gbest_score
            if improved_global:
                gbest_idx = np.argmin(pbest_scores)
                gbest, gbest_score = pbest[gbest_idx].copy(), pbest_scores[gbest_idx]

            evaluations += np.sum(update_indices)

        return gbest