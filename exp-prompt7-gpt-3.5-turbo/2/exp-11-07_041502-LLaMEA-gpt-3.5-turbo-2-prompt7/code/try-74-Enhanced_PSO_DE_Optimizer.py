import numpy as np

class Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        num_iters = int(budget / self.num_particles)
        self.bounds = (-5.0, 5.0)
        self.rng = np.random.default_rng()

    def __call__(self, func):
        swarm = self.rng.uniform(*self.bounds, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()

        for _ in range(num_iters):
            r_values = self.rng.random(size=(self.num_particles, 2))
            velocity = 0.5 * velocity + 2 * r_values[:, 0, None] * (pbest - swarm) + 2 * r_values[:, 1, None] * (gbest - swarm)
            swarm = np.clip(swarm + velocity, *self.bounds)
            current_scores = np.apply_along_axis(func, 1, swarm)

            updates = current_scores < pbest_scores
            pbest[updates] = swarm[updates]
            pbest_scores[updates] = current_scores[updates]

            gbest_idx = np.argmin(pbest_scores)
            gbest = pbest[gbest_idx].copy()

        return gbest