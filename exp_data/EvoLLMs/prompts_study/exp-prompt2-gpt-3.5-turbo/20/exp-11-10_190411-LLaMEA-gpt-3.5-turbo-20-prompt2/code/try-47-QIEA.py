import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Apply dynamic strategy for quantum rotation gates
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_angle = np.linspace(0, theta, num=self.dim)  # Dynamic angle adjustment
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)])
            population = np.dot(population, rotation_matrix)
            
            # Apply evolutionary operators
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution