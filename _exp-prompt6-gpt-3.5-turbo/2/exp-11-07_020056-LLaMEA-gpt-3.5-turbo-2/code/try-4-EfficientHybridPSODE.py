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

        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))

        population = initialize_particles()
        pbest = population.copy()
        gbest = population[np.argmin([fitness(p) for p in population])

        for _ in range(self.budget // self.num_particles):
            r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            velocity = self.w * (velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population))
            population = np.clip(population + velocity, -5.0, 5.0)

            for i in range(self.num_particles):
                a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = population[i] if np.random.rand() < self.cr else mutant
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial

            best_indices = np.argpartition([fitness(p) for p in population], self.num_particles)[:self.num_particles]
            best_particle = population[best_indices[0]]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest