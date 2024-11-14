import numpy as np

class NovelDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        
    def __call__(self, func):
        for _ in range(self.budget):
            # Mutation mechanism with adaptive strategies
            for i in range(self.budget):
                indices = [idx for idx in range(self.budget) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector
        best_solution = self.population[np.argmin([func(ind) for ind in self.population])]
        return best_solution