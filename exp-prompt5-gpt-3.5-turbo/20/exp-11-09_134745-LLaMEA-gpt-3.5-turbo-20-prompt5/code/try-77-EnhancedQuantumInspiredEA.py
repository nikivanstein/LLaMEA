import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def apply_mutation(self, individual, mutation_rate=0.1):
        mutated_individual = np.clip(individual + mutation_rate * np.random.normal(0, 1, self.dim), -5.0, 5.0)
        return mutated_individual

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                if idx != best_idx:
                    mutated_ind = self.apply_mutation(self.population[idx])
                    self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual) + mutated_ind)
        return best_individual