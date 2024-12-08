import numpy as np

class FastConvergingDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.crossover_rates = np.full(budget, 0.9)  # Initialize crossover rates

    def __call__(self, func):
        mutation_factor = 0.8
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling_factor = 0.8 + mutation_factor * np.std(self.population, axis=0)  # Dynamic mutation step size
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rates[i]  # Adaptive crossover rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
                    self.crossover_rates[i] = min(self.crossover_rates[i] * 1.02, 1.0)  # Adapt crossover rate based on performance
        return self.population[np.argmin([func(individual) for individual in self.population])]