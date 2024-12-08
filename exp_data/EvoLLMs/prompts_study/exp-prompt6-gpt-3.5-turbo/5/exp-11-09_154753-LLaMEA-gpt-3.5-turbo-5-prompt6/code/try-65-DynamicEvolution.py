import numpy as np

class DynamicEvolution(DifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.scaling_factors = np.full(budget, 0.8)

    def __call__(self, func):
        crossover_rate = 0.9
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling_factor = self.scaling_factors[i]
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
                    if scaling_factor < 1.0:
                        scaling_factor += 0.025  # Dynamically adjust scaling factor
                        self.scaling_factors[i] = min(scaling_factor, 1.0)
        return self.population[np.argmin([func(individual) for individual in self.population])]