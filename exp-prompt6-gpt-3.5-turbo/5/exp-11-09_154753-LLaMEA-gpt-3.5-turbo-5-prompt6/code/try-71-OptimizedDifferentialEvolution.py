import numpy as np

class OptimizedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        scaling_factor = 0.8
        crossover_rate = 0.9
        diversity_factor = 0.5
        for _ in range(self.budget):
            diversity = np.std(self.population)
            scaling_factor = 0.8 + diversity_factor * (diversity / 5.0)  # Dynamically adjust scaling factor
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
        return self.population[np.argmin([func(individual) for individual in self.population])]