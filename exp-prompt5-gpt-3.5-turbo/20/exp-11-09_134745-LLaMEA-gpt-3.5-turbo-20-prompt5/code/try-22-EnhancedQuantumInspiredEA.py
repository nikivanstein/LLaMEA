import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_mutation(self, individual, mutation_rate=0.1):
        mutated_individual = individual + np.random.normal(0, 1, size=self.dim) * mutation_rate
        return np.clip(mutated_individual, -5.0, 5.0)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                mutated_individual = self.apply_mutation(self.apply_gate(best_individual))
                self.population[idx] = 0.5 * (self.population[idx] + mutated_individual)
        return best_individual