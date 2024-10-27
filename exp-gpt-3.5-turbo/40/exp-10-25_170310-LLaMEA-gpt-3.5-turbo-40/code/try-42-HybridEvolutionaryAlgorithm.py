import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.scale_factor = 0.5
        self.crossover_prob = 0.9
        self.sa_temperature = 1.0
        self.sa_cooling_rate = 0.95

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, self.population[i])
                fitness_trial = func(trial)
                if fitness_trial < func(self.population[i]):
                    self.population[i] = trial
                else:
                    acceptance_prob = np.exp((func(self.population[i]) - fitness_trial) / self.sa_temperature)
                    if np.random.rand() < acceptance_prob:
                        self.population[i] = trial
            self.sa_temperature *= self.sa_cooling_rate

        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution