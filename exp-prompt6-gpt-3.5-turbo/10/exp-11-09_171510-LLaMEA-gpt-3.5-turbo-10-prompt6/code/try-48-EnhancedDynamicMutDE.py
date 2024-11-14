import numpy as np

class EnhancedDynamicMutDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.f_factors = np.random.uniform(0.5, 1.0, self.population_size)
        self.diversity_history = []

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.f_factors[i] * (self.population[b] - self.population[c])
                trial = np.where(np.logical_or(mutant < -5.0, mutant > 5.0), self.population[i], mutant)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                    self.f_factors[i] += 0.1 if self.f_factors[i] < 1.0 else -0.1

            diversity_metric = np.mean(np.std(self.population, axis=0))
            self.diversity_history.append(diversity_metric)

            if len(self.diversity_history) > 1 and self.diversity_history[-1] < self.diversity_history[-2]:
                for j in range(self.population_size):
                    self.f_factors[j] = min(1.0, self.f_factors[j] + 0.1)

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual