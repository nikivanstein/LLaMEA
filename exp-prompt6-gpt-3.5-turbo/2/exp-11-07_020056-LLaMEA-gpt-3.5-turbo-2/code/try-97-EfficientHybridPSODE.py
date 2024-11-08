import numpy as np

class EfficientHybridPSODE:
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
            velocity = self.w * velocity + self.c1 * np.random.uniform(0, 1, (self.num_particles, self.dim)) * (pbest - population) + self.c2 * np.random.uniform(0, 1, (self.num_particles, self.dim)) * (gbest - population)
            population = np.clip(population + velocity, -5.0, 5.0)

            for i in range(self.num_particles):
                a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = np.where(np.random.rand() < self.cr, mutant, population[i])
                population[i] = np.where(fitness(trial) < fitness(population[i]), trial, population[i])

            best_particle = population[np.argmin([fitness(p) for p in population])]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest