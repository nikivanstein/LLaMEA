import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Generate valid quantum rotation gates for each dimension
            thetas = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrices = np.array([[np.cos(thetas), -np.sin(thetas)], [np.sin(thetas), np.cos(thetas)]])
            
            population = np.dot(population, rotation_matrices)
            
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution