import numpy as np

class ImprovedHybridPSODE:
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

        def update_velocity(x, v, pbest, gbest):
            r = np.random.uniform(0, 1, (self.num_particles, 1))
            new_v = self.w * v + self.c1 * r * (pbest - x) + self.c2 * r * (gbest - x)
            return new_v

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        population = initialize_particles()
        best_particle = population[np.argmin([fitness(p) for p in population])]
        pbest = population.copy()
        gbest = best_particle.copy()

        for _ in range(self.budget // self.num_particles):
            velocity = update_velocity(population, population - pbest, pbest, gbest)
            r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            population = update_position(population, self.w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population))
            for i in range(self.num_particles):
                a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = population[i] if np.random.rand() < self.cr else mutant
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial

            best_particle = population[np.argmin([fitness(p) for p in population])]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest