import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim, mutation_rate=0.1):  # Changed to include mutation_rate
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate  # Added mutation_rate parameter
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual, mutation_rate):  # Added mutation_rate parameter
        return individual * np.random.choice([-1, 1], size=self.dim) * mutation_rate  # Updated to incorporate mutation_rate

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual, self.mutation_rate))  # Updated apply_gate call
        return best_individual