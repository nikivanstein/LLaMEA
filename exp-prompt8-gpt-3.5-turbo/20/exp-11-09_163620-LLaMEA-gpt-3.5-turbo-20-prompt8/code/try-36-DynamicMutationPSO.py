import numpy as np

class DynamicMutationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate_min = 0.01
        self.mutation_rate_max = 0.2
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        mutation_rate = self.mutation_rate_max

        for _ in range(self.budget):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            mutation_rate = self.mutation_rate_min + (_ / self.budget) * (self.mutation_rate_max - self.mutation_rate_min)
            mutation_mask = np.random.choice([0, 1], size=(self.dim, self.dim), p=[1 - mutation_rate, mutation_rate])
            swarm += mutation_mask * np.random.normal(0, 1, (self.dim, self.dim))

        return gbest_fitness