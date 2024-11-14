import numpy as np

class ImprovedChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                chaotic_map = np.sin(self.population[a]) * np.cos(self.population[b]) / (np.tanh(self.population[c]) + 1)
                scaling_factor = 0.5 + 0.5 * np.random.rand()
                levy_flight = np.random.standard_cauchy(self.dim)
                mutant = self.population[a] + scaling_factor * (self.population[b] - self.population[c]) + chaotic_map + levy_flight
                crossover_prob = 0.9 / (1 + np.exp(-10 * (func(mutant) - func(self.population[i]))))
                crossover = np.random.rand(self.dim) < crossover_prob
                trial = np.where(crossover, mutant, self.population[i])
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
        best_solution = self.population[np.argmin([func(x) for x in self.population])]
        return best_solution