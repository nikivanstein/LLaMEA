import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.1

    def adaptive_mutation(self, func, population):
        fitness_values = func(population)
        normalized_fitness = (fitness_values - np.min(fitness_values)) / (np.max(fitness_values) - np.min(fitness_values))
        self.mutation_scale = 0.1 + 0.4 * (1 - normalized_fitness)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            self.adaptive_mutation(func, population)
            # Apply quantum rotation gates to population
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply adaptive mutation
            offspring = population + np.random.normal(0, self.mutation_scale, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution