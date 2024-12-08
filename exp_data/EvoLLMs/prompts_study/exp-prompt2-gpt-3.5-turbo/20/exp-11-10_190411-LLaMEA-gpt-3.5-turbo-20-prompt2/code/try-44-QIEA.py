import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            thetas = np.random.uniform(-np.pi, np.pi, size=(self.budget, self.dim))
            rotation_matrices = np.array([[np.cos(thetas), -np.sin(thetas)], [np.sin(thetas), np.cos(thetas)])
            rotated_population = np.einsum('ijk,ik->ij', rotation_matrices, population)
            
            offspring = rotated_population + np.random.normal(0, 0.1, size=rotated_population.shape)
            population = np.where(func(offspring) < func(rotated_population), offspring, rotated_population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution