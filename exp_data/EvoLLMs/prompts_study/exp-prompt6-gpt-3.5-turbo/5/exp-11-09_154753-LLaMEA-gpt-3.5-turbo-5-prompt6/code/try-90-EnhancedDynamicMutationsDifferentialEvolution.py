import numpy as np

class EnhancedDynamicMutationsDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_factors = np.full(budget, 0.8)
        self.performance_history = np.zeros(budget)

    def __call__(self, func):
        crossover_rate = 0.9
        for _ in range(self.budget):
            diversity_factor = np.std(self.population, axis=0)
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling_factor = 0.8 + self.mutation_factors[i] * np.mean(diversity_factor)
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
                    self.mutation_factors[i] = min(self.mutation_factors[i] * 1.05, 5.0)
                    self.performance_history[i] += 1  # Update performance history for adaptive mutation
                else:
                    self.performance_history[i] = 0  # Reset performance counter if no improvement
            self.mutation_factors *= 0.95 + 0.05 * self.performance_history / np.max(self.performance_history)  # Adjust mutation factor based on individual performance
        return self.population[np.argmin([func(individual) for individual in self.population])]