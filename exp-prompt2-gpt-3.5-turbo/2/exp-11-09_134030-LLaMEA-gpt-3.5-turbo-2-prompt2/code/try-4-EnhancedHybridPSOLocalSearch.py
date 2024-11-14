import numpy as np

class EnhancedHybridPSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = 100
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.lb = -5.0
        self.ub = 5.0

    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        for _ in range(5):
            new_particle = np.clip(best_particle + 0.1 * np.random.randn(self.dim), self.lb, self.ub)
            if func(new_particle) < func(best_particle):
                best_particle = np.copy(new_particle)
        return best_particle

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        pbest = np.copy(population)
        pbest_fitness = np.array([func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = np.copy(pbest[gbest_idx])

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                w = self.w_max - (_ / self.max_iter) * (self.w_max - self.w_min)  # Dynamic inertia weight adaptation
                velocity = w * population[i] + self.c1 * r1 * (pbest[i] - population[i]) + self.c2 * r2 * (gbest - population[i])
                population[i] = np.clip(population[i] + velocity, self.lb, self.ub)
                population[i] = self.local_search(population[i], func)
                fitness = func(population[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = np.copy(population[i])
                    pbest_fitness[i] = fitness
                    if fitness < func(gbest):
                        gbest = np.copy(population[i])

        return gbest