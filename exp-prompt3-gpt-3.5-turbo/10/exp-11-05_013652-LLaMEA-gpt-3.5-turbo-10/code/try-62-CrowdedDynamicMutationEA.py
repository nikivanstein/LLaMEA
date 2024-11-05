import numpy as np

class CrowdedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        crowding_distance = np.zeros(self.budget)
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            crowding_distance[best_idx] = np.inf
            for i in range(self.budget):
                if i != best_idx:
                    crowding_distance[i] += np.linalg.norm(self.population[i] - best_solution)
            best_crowded_idx = np.argmax(crowding_distance)
            mutation_rate = np.random.uniform(0.01, 0.1)
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = self.population[best_crowded_idx] + mutation
            if func(new_solution) < fitness[best_crowded_idx]:
                self.population[best_crowded_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]