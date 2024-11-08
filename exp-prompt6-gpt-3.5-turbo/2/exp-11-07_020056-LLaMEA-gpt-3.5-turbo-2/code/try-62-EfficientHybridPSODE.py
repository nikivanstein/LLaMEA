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
        velocities = np.zeros((self.num_particles, self.dim))

        for _ in range(self.budget // self.num_particles):
            # Update velocities
            velocities = self.w * velocities + self.c1 * np.random.uniform(0, 1, (self.num_particles, self.dim)) * (population - population) + \
                         self.c2 * np.random.uniform(0, 1, (self.num_particles, self.dim)) * (population - population)

            # Update positions
            population = np.clip(population + velocities, -5.0, 5.0)

            # Differential evolution
            for i in range(self.num_particles):
                a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = population[i] if np.random.rand() < self.cr else mutant
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial

        return population[np.argmin([fitness(p) for p in population]])