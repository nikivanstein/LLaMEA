import numpy as np

class MultiPopulationDynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_populations = 5
        self.pop_size = int(budget / self.num_populations)
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        populations = [np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)) for _ in range(self.num_populations)]
        velocities = [np.zeros((self.pop_size, self.dim)) for _ in range(self.num_populations)]
        pbest = [pop.copy() for pop in populations]
        pbest_fitness = [np.apply_along_axis(func, 1, pop) for pop in pbest]
        gbest = [pop[np.argmin(fitness)] for pop, fitness in zip(pbest, pbest_fitness)]
        gbest_fitness = [np.min(fitness) for fitness in pbest_fitness]

        for _ in range(self.budget):
            for i in range(self.num_populations):
                swarm = populations[i]
                velocity = velocities[i]
                pbest_i = pbest[i]
                pbest_fitness_i = pbest_fitness[i]
                gbest_i = gbest[i]
                gbest_fitness_i = gbest_fitness[i]

                w = self.w_max

                r1 = np.random.random((self.pop_size, self.dim))
                r2 = np.random.random((self.pop_size, self.dim))

                velocity = w * velocity + self.c1 * r1 * (pbest_i - swarm) + self.c2 * r2 * (gbest_i - swarm)
                swarm += velocity

                fitness = np.apply_along_axis(func, 1, swarm)
                update_indices = fitness < pbest_fitness_i
                pbest_i[update_indices] = swarm[update_indices]
                pbest_fitness_i[update_indices] = fitness[update_indices]

                if np.min(fitness) < gbest_fitness_i:
                    gbest_i = swarm[np.argmin(fitness)]
                    gbest_fitness_i = np.min(fitness)

                w = self.w_min + (_ / self.budget) * (self.w_max - self.w_min)

        return np.mean(gbest_fitness)