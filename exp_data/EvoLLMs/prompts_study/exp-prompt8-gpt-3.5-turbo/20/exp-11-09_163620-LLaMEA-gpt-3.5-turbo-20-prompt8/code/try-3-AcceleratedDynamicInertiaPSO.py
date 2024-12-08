import numpy as np

class AcceleratedDynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.a = 0.5

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        for t in range(1, self.budget + 1):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            acceleration = self.a + 0.5 * (1 - t / self.budget)
            self.w = 0.4 + 0.5 * (1 - t / self.budget)

            velocity = self.w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm) * acceleration
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

        return gbest_fitness