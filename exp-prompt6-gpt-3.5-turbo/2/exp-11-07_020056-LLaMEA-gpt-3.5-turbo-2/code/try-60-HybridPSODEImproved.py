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

        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))

        def update_velocity(x, v, pbest, gbest):
            r = np.random.uniform(0, 1, (self.num_particles, self.dim, 2))
            new_v = self.w * v + r @ np.array([self.c1 * (pbest - x), self.c2 * (gbest - x)]).transpose(0, 2, 1)
            return new_v

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        def evolve_population(population):
            new_x = np.clip(population + population[:, None, self.num_particles:], -5.0, 5.0)
            new_x = np.where(fitness(new_x) < fitness(population), new_x, population)
            return new_x

        def differential_evolution(population):
            indexes = np.arange(self.num_particles)
            a, b, c = population[np.random.choice(np.delete(indexes, indexes[:, None], 3), 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.num_particles) < self.cr, mutant, population)
            trial = np.where(fitness(trial) < fitness(population), trial, population)
            return trial

        population = initialize_particles()
        best_particle = population[np.argmin([fitness(p) for p in population])]
        pbest = population.copy()
        gbest = best_particle.copy()

        for _ in range(self.budget // self.num_particles):
            velocity = update_velocity(population, population - pbest, pbest, gbest)
            population = evolve_population(population)
            population = differential_evolution(population)
            best_particle = population[np.argmin([fitness(p) for p in population])]
            pbest = np.where(fitness(population) < fitness(pbest), population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest