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

        def evolve_population(population):
            for _ in range(self.budget // self.num_particles):
                velocities = self.w * population + self.c1 * np.random.rand(self.num_particles, 1) * (population - population.min(axis=0)) + self.c2 * np.random.rand(1, self.dim) * (population.min(axis=0) - population)
                new_population = np.clip(population + velocities, -5.0, 5.0)
                fitness_values = np.apply_along_axis(fitness, 1, new_population)
                mask = fitness_values < np.apply_along_axis(fitness, 1, population)
                population[mask] = new_population[mask]

        population = initialize_particles()
        evolve_population(population)
        
        return population[np.argmin([fitness(p) for p in population])]