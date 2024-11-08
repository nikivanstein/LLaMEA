import numpy as np

class Dynamic_Inertia_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.5

    def __call__(self, func):
        swarm = np.random.uniform(*self.bounds, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocity[i] = self.inertia_weight * velocity[i] + 2 * r1 * (pbest[i] - swarm[i]) + 2 * r2 * (gbest - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], *self.bounds)
                
                current_score = func(swarm[i])
                if current_score < pbest_scores[i]:
                    pbest[i], pbest_scores[i] = swarm[i].copy(), current_score

                if current_score < pbest_scores[gbest_idx]:
                    gbest, gbest_idx = pbest[i].copy(), i

                self.inertia_weight = 0.4 + 0.5 * (1 - _ / self.max_iter)  # Dynamic inertia weight update

        return gbest