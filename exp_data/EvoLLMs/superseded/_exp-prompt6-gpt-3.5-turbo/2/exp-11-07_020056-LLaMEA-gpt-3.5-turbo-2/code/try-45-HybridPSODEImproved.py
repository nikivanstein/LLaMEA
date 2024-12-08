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

        def differential_evolution(population):
            for i in range(self.num_particles):
                a, b, c = population[np.random.choice(np.delete(np.arange(len(population)), i, 0), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = population[i] if np.random.rand() < self.cr else mutant
                if fitness(trial) < fitness(population[i]):
                    population[i] = trial
            return population

        population = initialize_particles()
        best_particle = population[np.argmin([fitness(p) for p in population])
        pbest = population.copy()
        gbest = best_particle.copy()

        for _ in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                velocity = self.w * population[i + self.num_particles] + self.c1 * np.random.uniform(0, 1, self.dim) * (pbest[i] - population[i]) + self.c2 * np.random.uniform(0, 1, self.dim) * (gbest - population[i])
                new_x = np.clip(population[i] + velocity, -5.0, 5.0)
                if fitness(new_x) < fitness(population[i]):
                    population[i] = new_x

            population = differential_evolution(population)
            best_particle = population[np.argmin([fitness(p) for p in population])]
            pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(population, pbest)], population, pbest)
            gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest

        return gbest