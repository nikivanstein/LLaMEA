import numpy as np

class Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        swarm = np.random.uniform(*self.bounds, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest_scores = np.apply_along_axis(func, 1, swarm)
        gbest_idx = np.argmin(pbest_scores)
        gbest = swarm[gbest_idx].copy()

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocity[i] = 0.5 * velocity[i] + 2 * r1 * (swarm[i] - swarm[i]) + 2 * r2 * (gbest - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], *self.bounds)

                current_score = func(swarm[i])
                if current_score < pbest_scores[i]:
                    pbest_scores[i] = current_score

                if current_score < pbest_scores[gbest_idx]:
                    gbest, gbest_idx = swarm[i].copy(), i

        return gbest