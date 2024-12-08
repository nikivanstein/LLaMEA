import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def diversity_maintenance(self):
        center = np.mean(self.population, axis=0)
        for idx in range(self.budget):
            self.population[idx] = 0.5 * (self.population[idx] + center)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            self.diversity_maintenance()
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
        return best_individual