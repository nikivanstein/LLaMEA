import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:2]]
            # Integrate local search for improved exploitation
            best_individual = self.population[np.argmin(fitness)]
            local_search = best_individual + 0.1 * np.random.normal(0, 1, self.dim)
            local_search_fitness = func(local_search)
            if local_search_fitness < fitness[np.argmax(fitness)]:
                self.population[np.argmax(fitness)] = local_search
            else:
                offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, 1, self.dim)
                worst_idx = np.argmax(fitness)
                self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]