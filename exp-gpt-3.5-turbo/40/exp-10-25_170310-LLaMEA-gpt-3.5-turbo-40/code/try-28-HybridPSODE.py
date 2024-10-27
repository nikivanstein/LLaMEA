import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.scale_factor = 0.5
        self.crossover_prob = 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                pbest = self.population[np.argmin([func(x) for x in self.population])]
                v = self.w * (self.population[i] - self.population[i]) + \
                    self.c1 * np.random.rand() * (pbest - self.population[i]) + \
                    self.c2 * np.random.rand() * (self.population[a] - self.population[b])
                x = self.population[i] + v
                mutant = x + self.scale_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, self.population[i])
                fitness_trial = func(trial)
                if fitness_trial < func(self.population[i]):
                    self.population[i] = trial
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution