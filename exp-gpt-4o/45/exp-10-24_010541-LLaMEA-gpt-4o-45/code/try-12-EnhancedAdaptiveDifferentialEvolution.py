import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)
        self.F = np.random.uniform(0.4, 0.9, self.pop_size)

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn() * sigma
        v = np.random.randn()
        step = u / abs(v) ** (1 / beta)
        return L * step

    def resize_population(self, evaluations):
        if evaluations < self.budget / 3:
            return self.pop_size
        elif evaluations < 2 * self.budget / 3:
            return self.pop_size // 2
        else:
            return self.pop_size // 4

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            current_pop_size = self.resize_population(evaluations)
            for i in range(current_pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with Lévy flight
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                L = self.levy_flight(1.0)
                mutant = self.population[a] + self.F[i] * (self.population[b] - self.population[c]) + L * (self.population[a] - self.population[i])
                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.9 * self.CR[i] + 0.1 * np.random.rand()
                    self.F[i] = 0.9 * self.F[i] + 0.1 * np.random.rand()
                else:
                    self.CR[i] = 0.1 * self.CR[i] + 0.9 * np.random.rand()
                    self.F[i] = 0.1 * self.F[i] + 0.9 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]