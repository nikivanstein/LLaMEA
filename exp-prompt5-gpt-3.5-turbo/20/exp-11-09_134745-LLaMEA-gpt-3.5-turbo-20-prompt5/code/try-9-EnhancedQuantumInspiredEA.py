import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_mutation(self, individual, mutation_rate=0.1):
        mutation_mask = np.random.choice([True, False], size=self.dim, p=[mutation_rate, 1 - mutation_rate])
        return individual + np.random.uniform(-1.0, 1.0, size=self.dim) * mutation_mask

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
                self.population[idx] = self.apply_mutation(self.population[idx], mutation_rate=0.1)  # Integrate mutation
        return best_individual