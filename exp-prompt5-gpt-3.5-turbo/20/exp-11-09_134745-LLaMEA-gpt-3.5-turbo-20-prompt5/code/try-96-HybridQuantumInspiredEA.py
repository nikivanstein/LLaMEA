import numpy as np

class HybridQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def local_search(self, individual, func):
        perturbed_individual = individual + np.random.uniform(-0.5, 0.5, self.dim)
        if func(perturbed_individual) < func(individual):
            return perturbed_individual
        else:
            return individual

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
                self.population[idx] = self.local_search(self.population[idx], func)
        return best_individual