import numpy as np

class EnhancedOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(dim, 0.5)
        self.population_size = budget

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            
            for i in range(self.dim):
                mutation_rate = np.clip(self.mutation_rates[i] + np.random.normal(0, 0.1), 0.1, 0.9)
                population[:, i] = best_individual[i] + mutation_rate * np.random.standard_normal(self.population_size)
            
            fitness = np.array([func(individual) for individual in population])

            # Dynamic population size adjustment based on individual performance
            good_individuals = np.sum(fitness < np.mean(fitness))
            bad_individuals = self.population_size - good_individuals
            self.population_size = int(self.population_size + 0.1 * (good_individuals - bad_individuals))
            population = np.vstack((population[sorted_indices[:good_individuals]], np.random.uniform(-5.0, 5.0, (bad_individuals, self.dim))))
            fitness = np.array([func(individual) for individual in population])
        
        return best_individual