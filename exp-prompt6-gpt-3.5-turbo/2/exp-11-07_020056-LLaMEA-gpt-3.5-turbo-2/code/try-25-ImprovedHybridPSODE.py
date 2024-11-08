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
            r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
            new_v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            return new_v

        def update_position(x, v):
            return np.clip(x + v, -5.0, 5.0)

        population = initialize_particles()
        pbest = population.copy()
        gbest = population[np.argmin([fitness(p) for p in population])

        for _ in range(self.budget // self.num_particles):
            velocity = update_velocity(population, population - pbest, pbest, gbest)
            for i in range(self.num_particles):
                new_x = update_position(population[i], velocity[i + self.num_particles])
                if fitness(new_x) < fitness(population[i]):
                    population[i] = new_x
                    pbest[i] = new_x if fitness(new_x) < fitness(pbest[i]) else pbest[i]
                    gbest = new_x if fitness(new_x) < fitness(gbest) else gbest

        return gbest