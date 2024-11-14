import numpy as np

class FasterConvergingDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_factors = np.full(budget, 0.8)
        self.global_best = self.population[np.argmin([func(individual) for individual in self.population])]

    def __call__(self, func):
        crossover_rate = 0.9
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling_factor = 0.8 + self.mutation_factors[i] + np.linalg.norm(self.global_best - self.population[i]) * 0.01
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
                    self.mutation_factors[i] = min(self.mutation_factors[i] * 1.05, 5.0)  # Adaptive mutation step size
        return self.population[np.argmin([func(individual) for individual in self.population])]