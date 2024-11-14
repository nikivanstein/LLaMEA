import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        swarm_size = 30
        max_iter = int(self.budget / swarm_size)
        c1 = 2.0
        c2 = 2.0
        w = 0.5
        f = 0.5

        def DE(p, gbest):
            r1, r2, r3 = np.random.choice(swarm_size, 3, replace=False)
            mutant = p[r1] + f * (p[r2] - p[r3])
            mutant = np.clip(mutant, -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) < 0.5, mutant, p)
            trial_fitness = func(trial)
            if trial_fitness < func(p):
                return trial
            else:
                return p

        population = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
        velocity = np.zeros((swarm_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(p) for p in population])
        gbest = population[np.argmin(pbest_fitness)]

        for _ in range(max_iter):
            for i in range(swarm_size):
                velocity[i] = w * velocity[i] + c1 * np.random.rand(self.dim) * (pbest[i] - population[i]) + c2 * np.random.rand(self.dim) * (gbest - population[i])
                population[i] = np.clip(population[i] + velocity[i], -5.0, 5.0)
                population[i] = DE(population[i], gbest)
                fitness = func(population[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = fitness
                if fitness < func(gbest):
                    gbest = population[i]

        return gbest