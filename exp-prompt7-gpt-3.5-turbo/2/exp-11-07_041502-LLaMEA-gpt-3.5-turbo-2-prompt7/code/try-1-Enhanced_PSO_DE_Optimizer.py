import numpy as np

class Enhanced_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def fitness(x):
            return func(x)

        swarm = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_scores = np.array([fitness(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        
        for _ in range(self.max_iter):
            r1, r2 = np.random.uniform(size=(2, self.num_particles, self.dim))
            velocity = 0.5 * velocity + 2 * r1 * (pbest - swarm) + 2 * r2 * (gbest - swarm)
            swarm += velocity
            np.clip(swarm, self.lower_bound, self.upper_bound, out=swarm)
            
            updated = fitness(swarm) < fitness(pbest)
            pbest[updated] = swarm[updated]
            updated = fitness(pbest) < fitness(gbest)
            gbest = np.where(updated, pbest, gbest)

        return gbest