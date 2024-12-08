import numpy as np

class EnhancedQuantumInspiredEA(QuantumInspiredEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.1

    def adaptive_mutation_rate(self, fitness):
        self.mutation_rate = np.clip(0.1 / np.mean(fitness), 0.01, 0.5)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            self.adaptive_mutation_rate(fitness)
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.mutation_rate * self.apply_gate(best_individual))
        return best_individual