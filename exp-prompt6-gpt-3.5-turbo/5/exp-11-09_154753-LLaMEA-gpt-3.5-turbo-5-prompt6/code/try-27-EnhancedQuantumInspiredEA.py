import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        crowding_distances = np.zeros(self.budget)
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents_indices = np.argsort(fitness)[:2]
            parents = self.population[parents_indices]
            niche_count = [np.sum(np.abs(self.population - self.population[i]) < 1.0, axis=1) for i in range(self.budget)]
            crowding_distances = np.sum(niche_count, axis=0)
            offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, 1, self.dim)
            worst_idx = parents_indices[np.argmax(crowding_distances[parents_indices])]
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]