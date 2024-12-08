import numpy as np
from scipy.spatial.distance import cdist

class CrowdedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rate = np.random.uniform(0.01, 0.1)
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            
            crowding_distances = np.mean(cdist(self.population, self.population), axis=0)
            diversity_bonus = 0.1  # Adjust the diversity bonus factor
            diversity_scores = np.exp(-crowding_distances/diversity_bonus)
            new_solution += np.sum(self.population * diversity_scores[:, np.newaxis], axis=0)
            
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]