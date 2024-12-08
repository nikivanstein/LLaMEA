import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def mutation(self, individual, mutation_rate=0.1):
        return individual + np.random.uniform(-0.5, 0.5, size=self.dim) if np.random.rand() < mutation_rate else individual

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                mutated_individual = self.mutation(0.5 * (self.population[idx] + self.apply_gate(best_individual)))
                self.population[idx] = mutated_individual
        return best_individual