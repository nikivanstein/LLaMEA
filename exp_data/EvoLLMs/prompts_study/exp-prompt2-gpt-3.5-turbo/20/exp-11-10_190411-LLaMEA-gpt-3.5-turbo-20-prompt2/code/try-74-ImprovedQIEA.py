import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Apply quantum rotation gates to population
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply new mutation operator
            mutation_factor = np.random.normal(0, 0.1, size=population.shape)
            population += mutation_factor
            
            # Apply evolutionary operators with elitism
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            new_population = np.where(func(offspring) < func(population), offspring, population)
            new_population[np.argmax(func(population))] = population[np.argmin(func(population))]

            current_best_solution = new_population[np.argmin(func(new_population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution