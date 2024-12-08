import numpy as np

class RefinedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.scale_factor = 0.5

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_mutation(self, individual):
        return individual + np.random.standard_cauchy(size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = self.scale_factor * (self.population[idx] + self.apply_gate(best_individual))
                self.population[idx] = self.apply_mutation(self.population[idx])
        return best_individual