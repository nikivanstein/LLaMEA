import numpy as np

class HybridQuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def differential_evolution(self, best_individual, idx):
        r1, r2, r3 = np.random.choice(self.population, 3, replace=False)
        mutant = r1 + 0.8 * (r2 - r3)
        return mutant if np.random.rand() < 0.9 else best_individual

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.differential_evolution(best_individual, idx) + self.apply_gate(best_individual))
        return best_individual