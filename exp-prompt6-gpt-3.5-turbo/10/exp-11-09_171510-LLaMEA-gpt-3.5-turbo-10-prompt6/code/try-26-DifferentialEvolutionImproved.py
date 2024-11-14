import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.f_factor = 0.8
        self.cr = 0.9  # Introducing crossover probability

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.f_factor * (self.population[b] - self.population[c])
                trial = np.where(np.logical_or(mutant < -5.0, mutant > 5.0), self.population[i], mutant)
                
                # Introducing crossover operation
                mask = np.random.rand(self.dim) < self.cr
                trial = np.where(mask, trial, self.population[i])

                if func(trial) < func(self.population[i]):
                    self.population[i] = trial

            self.f_factor *= 0.95
            self.cr = max(0.1, self.cr * 0.95)  # Adaptive update of crossover probability

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual