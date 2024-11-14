import numpy as np

class IQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Apply quantum rotation gates with dynamic learning rate
            learning_rate = 1.0 - _ / self.budget  # Dynamic learning rate
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(learning_rate*theta), -np.sin(learning_rate*theta)], 
                                        [np.sin(learning_rate*theta), np.cos(learning_rate*theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply evolutionary operators
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution