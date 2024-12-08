import numpy as np

class EnhancedEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.scale_factor = 0.5
        self.crossover_prob = 0.9
        self.line_refine_prob = 0.35

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, self.population[i])
                
                if np.random.rand() < self.line_refine_prob:
                    for j in range(self.dim):
                        trial[j] = np.clip(trial[j] + np.random.normal(0, 0.1), -5, 5)
                
                fitness_trial = func(trial)
                if fitness_trial < func(self.population[i]):
                    self.population[i] = trial
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution