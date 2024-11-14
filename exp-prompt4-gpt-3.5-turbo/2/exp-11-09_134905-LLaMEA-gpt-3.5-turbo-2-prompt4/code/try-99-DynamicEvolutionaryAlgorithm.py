import numpy as np

class DynamicEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.mutation_rate = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.apply_along_axis(func, 1, population)
        
        for _ in range(self.budget // self.population_size):
            parents = population[np.argsort(fitness_values)[:2]]
            offspring = parents[0] + np.random.uniform(-1, 1, parents.shape) * self.mutation_rate
            offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
            offspring_fitness = np.apply_along_axis(func, 1, offspring)
            
            worst_idx = np.argmax(fitness_values)
            if offspring_fitness[0] < fitness_values[worst_idx]:
                population[worst_idx] = offspring[0]
                fitness_values[worst_idx] = offspring_fitness[0]
            
        best_idx = np.argmin(fitness_values)
        return population[best_idx]