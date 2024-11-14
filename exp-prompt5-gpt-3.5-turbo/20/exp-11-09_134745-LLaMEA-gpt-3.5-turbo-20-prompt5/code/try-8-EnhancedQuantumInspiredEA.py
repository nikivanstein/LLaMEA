import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_mutation(self, individual):
        return individual + np.random.normal(0, 0.5, size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                if np.random.rand() < 0.1:  # Mutation probability
                    self.population[idx] = self.apply_mutation(0.5 * (self.population[idx] + self.apply_gate(best_individual)))
                else:
                    self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
        return best_individual