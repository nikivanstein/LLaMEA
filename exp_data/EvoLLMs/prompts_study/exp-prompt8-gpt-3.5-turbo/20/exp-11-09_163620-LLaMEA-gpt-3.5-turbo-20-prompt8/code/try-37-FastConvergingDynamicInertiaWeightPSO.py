import numpy as np

class FastConvergingDynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c_min = 1.5
        self.c_max = 2.5

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        w = self.w_max
        c1 = self.c_max
        c2 = self.c_max

        for _ in range(self.budget):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = w * velocity + c1 * r1 * (pbest - swarm) + c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            w = self.w_min + (_ / self.budget) * (self.w_max - self.w_min)
            c1 = self.c_min + (_ / self.budget) * (self.c_max - self.c_min)
            c2 = self.c_min + (_ / self.budget) * (self.c_max - self.c_min)

        return gbest_fitness