import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Adaptively adjust rotation angles based on function evaluations
            self.theta += np.random.normal(0, 0.1, size=self.dim)
            rotation_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply evolutionary operators
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution