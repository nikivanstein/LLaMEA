import numpy as np

class PopulationBasedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def optimize(self, func):
        evaluations = 0
        while evaluations < self.budget:
            fitness_values = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness_values)
            best_individual = self.population[sorted_indices[0]]
            new_population = [best_individual + np.random.normal(0, 0.1, self.dim) for _ in range(self.population_size)]
            self.population = new_population
            evaluations += self.population_size
        return best_individual