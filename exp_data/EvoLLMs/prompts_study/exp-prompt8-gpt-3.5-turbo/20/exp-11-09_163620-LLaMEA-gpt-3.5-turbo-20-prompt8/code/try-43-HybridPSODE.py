import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.F = 0.5
        self.CR = 0.7

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        w = self.w_max

        for _ in range(self.budget):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            for i in range(self.dim):
                trial_vector = swarm[i]
                idxs = np.random.choice(self.dim, 3, replace=False)
                diff_vector = swarm[idxs[0]] + self.F * (swarm[idxs[1]] - swarm[idxs[2]])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_mask] = diff_vector[crossover_mask]

                trial_fitness = func(trial_vector)
                if trial_fitness < pbest_fitness[i]:
                    pbest[i] = trial_vector
                    pbest_fitness[i] = trial_fitness
                    if trial_fitness < gbest_fitness:
                        gbest = trial_vector
                        gbest_fitness = trial_fitness

            w = self.w_min + (_ / self.budget) * (self.w_max - self.w_min)

        return gbest_fitness