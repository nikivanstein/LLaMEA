# import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def crowding_distance(self, fitness):
        sorted_indices = np.argsort(fitness)
        dist = np.zeros(len(fitness))
        dist[sorted_indices[0]] = dist[sorted_indices[-1]] = np.inf
        for i in range(1, len(fitness) - 1):
            dist[sorted_indices[i]] += (fitness[sorted_indices[i + 1]] - fitness[sorted_indices[i - 1]])
        return dist

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            crowding_dist = self.crowding_distance(fitness)
            best_idx = np.argmin(crowding_dist)
            best_individual = self.population[best_idx]
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
        return best_individual