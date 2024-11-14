import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def differential_evolution(self, idx, scaling_factor=0.8, crossover_prob=0.9):
        a, b, c = np.random.choice(self.population, 3, replace=False)
        mutant = a + scaling_factor * (b - c)
        crossover_mask = np.random.rand(self.dim) < crossover_prob
        trial = np.where(crossover_mask, mutant, self.population[idx])
        return trial

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                trial = self.differential_evolution(idx)
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual) + trial)
        return best_individual