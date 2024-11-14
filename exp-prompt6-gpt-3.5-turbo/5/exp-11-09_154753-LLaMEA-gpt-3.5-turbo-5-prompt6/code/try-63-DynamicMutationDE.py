import numpy as np

class DynamicMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        scaling_factors = np.linspace(0.5, 1.0, self.budget)  # Dynamic scaling factor
        crossover_rate = 0.9
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant_vector = self.population[a] + scaling_factors[i] * (self.population[b] - self.population[c])  # Dynamic scaling
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
        return self.population[np.argmin([func(individual) for individual in self.population])]