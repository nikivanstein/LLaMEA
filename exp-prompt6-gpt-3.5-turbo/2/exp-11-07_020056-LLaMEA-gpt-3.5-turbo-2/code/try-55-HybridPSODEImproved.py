import numpy as np

class HybridPSODEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.cr = 0.5
        self.f = 0.8

    def __call__(self, func):
        def fitness(x):
            return func(x)

        population = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        pbest = population.copy()
        gbest = population[np.argmin([fitness(p) for p in population])

        for _ in range(self.budget // self.num_particles):
            r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            velocity = self.w * (population - pbest) + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocity, -5.0, 5.0)

            idx = np.arange(self.num_particles)
            np.random.shuffle(idx)
            for i in idx:
                a, b, c = population[np.random.choice(np.delete(idx, i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = mutant if np.random.rand() < self.cr else population[i]
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial

            best_idxs = np.argmin([fitness(p) for p in population])
            if fitness(population[best_idxs]) < fitness(gbest):
                gbest = population[best_idxs]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)

        return gbest