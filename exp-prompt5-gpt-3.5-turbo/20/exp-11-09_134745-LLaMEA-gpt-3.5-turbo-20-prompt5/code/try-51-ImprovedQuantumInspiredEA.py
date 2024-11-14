import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def mutate(self, individual, mutation_rate=0.1):
        mutation_indices = np.random.choice([True, False], size=self.dim, p=[mutation_rate, 1 - mutation_rate])
        individual[mutation_indices] = np.random.uniform(-5.0, 5.0, np.sum(mutation_indices))
        return individual

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                mutated_individual = self.mutate(0.5 * (self.population[idx] + self.apply_gate(best_individual)))
                self.population[idx] = 0.5 * (self.population[idx] + mutated_individual)
        return best_individual