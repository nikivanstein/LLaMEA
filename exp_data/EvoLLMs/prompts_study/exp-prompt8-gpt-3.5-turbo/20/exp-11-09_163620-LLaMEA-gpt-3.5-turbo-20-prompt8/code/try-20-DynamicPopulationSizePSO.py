import numpy as np

class DynamicPopulationSizePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        num_particles = self.dim
        swarm = np.random.uniform(-5.0, 5.0, (num_particles, self.dim))
        velocity = np.zeros((num_particles, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        w = self.w_max

        for _ in range(self.budget):
            r1 = np.random.random((num_particles, self.dim))
            r2 = np.random.random((num_particles, self.dim))

            velocity = w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            w = self.w_min + (_ / self.budget) * (self.w_max - self.w_min)

            if np.random.rand() < 0.5 and num_particles > 2:
                num_particles -= 1
                swarm = np.vstack((swarm, np.random.uniform(-5.0, 5.0, (1, self.dim))))
                velocity = np.vstack((velocity, np.zeros((1, self.dim)))
                pbest = np.vstack((pbest, swarm[-1]))
                pbest_fitness = np.append(pbest_fitness, func(pbest[-1]))

        return gbest_fitness