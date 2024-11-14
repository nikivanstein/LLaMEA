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

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        population = initialize_particles()
        best_particle = population[np.argmin([fitness(p) for p in population])
        pbest = population.copy()
        gbest = best_particle.copy()

        for _ in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                new_v = self.w * velocity[i] + self.c1 * r1 * (pbest[i] - population[i]) + self.c2 * r2 * (gbest - population[i])
                new_x = update_position(population[i], new_v)
                if fitness(new_x) < fitness(population[i]):
                    population[i] = new_x
                    pbest[i] = new_x
                    if fitness(new_x) < fitness(gbest):
                        gbest = new_x

        return gbest