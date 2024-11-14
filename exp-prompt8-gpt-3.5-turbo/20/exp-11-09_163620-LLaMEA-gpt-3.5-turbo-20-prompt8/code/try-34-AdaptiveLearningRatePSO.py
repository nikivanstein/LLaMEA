import numpy as np

class AdaptiveLearningRatePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.c1_min = 1.5
        self.c1_max = 2.5
        self.c2_min = 1.5
        self.c2_max = 2.5

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        c1 = np.random.uniform(self.c1_min, self.c1_max)
        c2 = np.random.uniform(self.c2_min, self.c2_max)

        for _ in range(self.budget):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = velocity + c1 * r1 * (pbest - swarm) + c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            c1 = self.c1_min + (_ / self.budget) * (self.c1_max - self.c1_min)
            c2 = self.c2_min + (_ / self.budget) * (self.c2_max - self.c2_min)

        return gbest_fitness