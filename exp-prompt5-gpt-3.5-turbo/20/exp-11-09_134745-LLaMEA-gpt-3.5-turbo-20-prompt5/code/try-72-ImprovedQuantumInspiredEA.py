import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_dynamic_gate(self, individual, best_individual):
        gate_prob = np.clip(1 / (1 + np.exp(-np.linalg.norm(individual - best_individual))), 0.2, 0.8)
        gate = np.random.choice([-1, 1], size=self.dim, p=[gate_prob, 1 - gate_prob])
        return individual * gate

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_dynamic_gate(self.population[idx], best_individual))
        return best_individual