import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def differential_evolution(self, current, target, mutation_factor=0.5):
        return current + mutation_factor * (target - current)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                target_individual = self.population[np.random.choice(np.arange(self.budget))]
                self.population[idx] = self.differential_evolution(self.population[idx], target_individual)
        return best_individual