import numpy as np

class AdaptiveParticle_Evolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.alpha = 0.5
        self.beta = 0.9
        
    def adaptive_parameters(self, fitness):
        norm_fit = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        mutation_factor = 0.5 + 0.5 * (1 - norm_fit)
        return mutation_factor

    def __call__(self, func):
        pbest = self.pop.copy()
        pbest_fitness = self.fitness.copy()
        gbest_idx = np.argmin(self.fitness)
        gbest = self.pop[gbest_idx]

        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            self.func_evals += 1
            if self.func_evals >= self.budget:
                return self.pop[np.argmin(self.fitness)]

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.pop[np.argmin(self.fitness)]

                mutation_factor = self.adaptive_parameters(self.fitness)
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.alpha * self.velocities[i] +
                                      r1 * (pbest[i] - self.pop[i]) +
                                      r2 * (gbest - self.pop[i]))
                self.pop[i] += self.velocities[i]
                self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)

                if np.random.rand() < mutation_factor[i]:
                    dim_idx = np.random.randint(self.dim)
                    self.pop[i, dim_idx] = np.random.uniform(self.lb, self.ub)

                f_new = func(self.pop[i])
                self.func_evals += 1
                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    pbest[i] = self.pop[i]
                    pbest_fitness[i] = f_new
                    if f_new < self.fitness[gbest_idx]:
                        gbest_idx = i
                        gbest = self.pop[i]

        return self.pop[np.argmin(self.fitness)]