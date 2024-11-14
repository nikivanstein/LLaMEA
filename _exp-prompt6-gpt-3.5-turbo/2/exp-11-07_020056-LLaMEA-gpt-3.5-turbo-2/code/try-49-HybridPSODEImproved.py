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
            pbest = population.copy()
            best_particle = population[np.argmin([fitness(p) for p in population])
            gbest = best_particle.copy()
            for _ in range(self.budget // self.num_particles):
                r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
                r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
                velocity = self.w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
                new_population = np.clip(population + velocity, -5.0, 5.0)
                for i in range(self.num_particles):
                    if fitness(new_population[i]) < fitness(population[i]):
                        population[i] = new_population[i]
                        pbest[i] = new_population[i]
                for i in range(self.num_particles):
                    a, b, c = new_population[np.random.choice(np.delete(np.arange(len(new_population)), i, 0), 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                    trial = new_population[i] if np.random.rand() < self.cr else mutant
                    if fitness(trial) < fitness(new_population[i]):
                        new_population[i] = trial
                best_particle = new_population[np.argmin([fitness(p) for p in new_population])]
                pbest = np.where([fitness(p) < fitness(pb) for p, pb in zip(new_population, pbest)], new_population, pbest)
                gbest = best_particle if fitness(best_particle) < fitness(gbest) else gbest
            return gbest

        population = initialize_particles()
        return evolve_population(population)