import numpy as np

class EnhancedHybridPSODE:
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
            r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            new_v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            return new_v

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        def evolve_population(population):
            new_positions = update_position(population, population + self.num_particles)
            improvements = fitness(new_positions) < fitness(population)
            population[improvements] = new_positions[improvements]
            return population

        def differential_evolution(population):
            a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), np.arange(self.num_particles), 1), (self.num_particles, 3), replace=False].T
            mutants = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trials = np.where(np.random.rand(self.num_particles)[:, np.newaxis] < self.cr, mutants, population)
            improvements = fitness(trials) < fitness(population)
            population[improvements] = trials[improvements]
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
            improvements = fitness(best_particle) < fitness(gbest)
            gbest = np.where(improvements, best_particle, gbest)

        return gbest