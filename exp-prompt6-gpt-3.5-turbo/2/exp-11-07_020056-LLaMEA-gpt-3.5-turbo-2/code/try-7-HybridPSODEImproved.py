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
            new_v = self.w * v[:, np.newaxis] + np.array([[[self.c1, self.c2]]]) * r * (pbest - x)[:, np.newaxis]
            return new_v.sum(axis=2)

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        def evolve_population(population):
            for i in range(self.num_particles):
                new_x = update_position(population[i], population[i + self.num_particles])
                if fitness(new_x) < fitness(population[i]):
                    population[i] = new_x
            return population

        def differential_evolution(population):
            for i in range(self.num_particles):
                abc_indices = np.delete(np.arange(len(population)), i, 0)
                abc = population[np.random.choice(abc_indices, 3, replace=False)]
                mutant = np.clip(abc[:, 0] + self.f * (abc[:, 1] - abc[:, 2]), -5.0, 5.0)
                trial = np.where(np.random.rand() < self.cr, mutant, population[i])
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial
            return population

        population = initialize_particles()
        best_particle = population[np.argmin([fitness(p) for p in population])]
        pbest = population.copy()
        gbest = best_particle.copy()

        for _ in range(self.budget // self.num_particles):
            velocity = update_velocity(population, population - pbest, pbest, gbest)
            population = evolve_population(population)
            population = differential_evolution(population)
            best_particle = population[np.argmin([fitness(p) for p in population])]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest